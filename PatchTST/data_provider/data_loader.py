#Working Data Loader

from vmdpy import VMD  # ensure vmdpy is installed: pip install vmdpy
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from PatchTST.utils.time_features import time_features
import warnings
warnings.filterwarnings('ignore')

from joblib import load
import os

def load_imf_scalers(scaler_dir, num_imfs):
    scalers = {}
    for i in range(num_imfs):
        scaler_path = os.path.join(scaler_dir, f"hello{i}.pkl")
        scalers[i] = load(scaler_path)
    return scalers

# Example usage

def _run_vmd(signal,
             alpha=2000, tau=0., K=5, DC=0, init=1, tol=1e-7):
    """
    Run VMD on a 1D numpy array signal and return modes as shape (K, N).
    Wrap vmdpy.VMD for convenience.
    """
    # vmdpy.VMD returns (u, u_hat, omega) where u is modes (K, N)
    u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
    u = np.asarray(u)  # shape (K, N)
    # If VMD returns fewer modes pad with zeros (defensive)
    if u.shape[0] < K:
        pad = np.zeros((K - u.shape[0], u.shape[1]))
        u = np.vstack([u, pad])
    return u


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 use_vmd=True, num_imfs=5, scaler_type='minmax',
                 vmd_alpha=2000, vmd_tau=0., vmd_DC=0, vmd_init=1, vmd_tol=1e-7):
        
        # Sequence lengths
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 1
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_vmd = use_vmd
        self.num_imfs = int(num_imfs)
        self.scaler_type = scaler_type.lower()

        # VMD parameters
        self.vmd_alpha = vmd_alpha
        self.vmd_tau = vmd_tau
        self.vmd_DC = vmd_DC
        self.vmd_init = vmd_init
        self.vmd_tol = vmd_tol

        # ---- STEP 1: Read raw data ----
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if 'date' not in df_raw.columns or self.target not in df_raw.columns:
            raise ValueError("CSV must contain 'date' and target column")

        self.dates = pd.to_datetime(df_raw['date'])
        self.raw_signal = df_raw[self.target].values.astype(float)

        # ---- STEP 2: Train/Val/Test Split ----
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_val = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        self.border1, self.border2 = border1s[self.set_type], border2s[self.set_type]

        # ---- STEP 3: Create time encodings ----
        df_stamp = pd.DataFrame({'date': self.dates[self.border1:self.border2]})
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            self.data_stamp = df_stamp.drop('date', axis=1).values
        else:
            self.data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq).transpose(1, 0)

    def __getitem__(self, index):
        # ---- STEP 1: Define input and label windows ----
        s_begin = self.border1 + index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len  # label_len + pred_len

        # Input and output segments from raw data
        window_in = self.raw_signal[s_begin:s_end]            # for encoder
        window_out = self.raw_signal[r_begin:r_end]           # for decoder

        # ---- STEP 2: Decompose using VMD ----
        imfs_in = _run_vmd(window_in,
                           alpha=self.vmd_alpha, tau=self.vmd_tau,
                           K=self.num_imfs, DC=self.vmd_DC,
                           init=self.vmd_init, tol=self.vmd_tol)
        imfs_out = _run_vmd(window_out,
                            alpha=self.vmd_alpha, tau=self.vmd_tau,
                            K=self.num_imfs, DC=self.vmd_DC,
                            init=self.vmd_init, tol=self.vmd_tol)

        # ---- STEP 3: Scale both using SAME scaler per IMF ----
        if self.scale and self.use_vmd:
            scaled_in = np.zeros_like(imfs_in)
            scaled_out = np.zeros_like(imfs_out)
            for i in range(self.num_imfs):
                sc = MinMaxScaler((0, 1)) if self.scaler_type == 'minmax' else StandardScaler()
                sc.fit(imfs_in[i].reshape(-1, 1))  # Fit on input IMF
                scaled_in[i] = sc.transform(imfs_in[i].reshape(-1, 1)).reshape(-1)
                scaled_out[i] = sc.transform(imfs_out[i].reshape(-1, 1)).reshape(-1)
            imfs_in, imfs_out = scaled_in, scaled_out

        # ---- STEP 4: Prepare tensors ----
        seq_x = torch.tensor(imfs_in, dtype=torch.float).T                     # (num_imfs, seq_len)
        seq_y = torch.tensor(imfs_out[:, -self.pred_len:], dtype=torch.float).T                    # (num_imfs, label_len + pred_len)
        seq_x_mark = torch.tensor(self.data_stamp[s_begin:s_end], dtype=torch.float)
        seq_y_mark = torch.tensor(self.data_stamp[s_end:r_end], dtype=torch.float)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.border2 - self.border1 - self.seq_len - self.pred_len

    def inverse_transform_imfs(self, imf_preds):
        """
        Inverse-transform per-IMF scaled predictions and sum them to reconstruct the final target series.

        imf_preds expected shapes:
          - (K, T)  : K imfs, T time steps
          - (B, K, T) : batch, K, T
          - (B, T, K) : batch, T, K  (will transpose)
        Returns:
          - reconstructed array shape matching non-batch or (B, T)
        """
        if not self.use_vmd:
            # fallback to single-scaler inverse
            if self.scaler is None:
                raise RuntimeError("No scaler available for inverse transform")
            arr = np.asarray(imf_preds)
            # if arr is batch or 2D, flatten last dims and inverse scalers as needed
            # Assume last dim is feature dim for single-scaler case
            shape = arr.shape
            flat = arr.reshape(-1, shape[-1]) if arr.ndim >= 2 else arr.reshape(-1, 1)
            inv = self.scaler.inverse_transform(flat)
            return inv.reshape(shape)
        # use_vmd True: process per-imf
        K = self.num_imfs
        arr = np.asarray(imf_preds)
        # unify to (B?, K, T)
        if arr.ndim == 2 and arr.shape[0] == K:
            # (K, T) -> (K, T)
            k_t = arr
            # inverse per imf
            inv_imfs = []
            for i in range(K):
                sc = self.scalers[i]
                col = k_t[i].reshape(-1, 1)
                inv_col = sc.inverse_transform(col).reshape(-1)
                inv_imfs.append(inv_col)
            inv_imfs = np.vstack(inv_imfs)  # (K, T)
            # sum across imfs -> (T,)
            return inv_imfs.sum(axis=0)
        elif arr.ndim == 3:
            # could be (B, K, T) or (B, T, K)
            if arr.shape[1] == K:
                # assume (B, K, T)
                B, K_, T = arr.shape
                out = np.zeros((B, T))
                for b in range(B):
                    inv_imfs = []
                    for i in range(K):
                        sc = self.scalers[i]
                        col = arr[b, i, :].reshape(-1, 1)
                        inv_col = sc.inverse_transform(col).reshape(-1)
                        inv_imfs.append(inv_col)
                    inv_imfs = np.vstack(inv_imfs)  # (K, T)
                    out[b] = inv_imfs.sum(axis=0)
                return out
            elif arr.shape[2] == K:
                # (B, T, K) -> transpose to (B, K, T)
                arr2 = arr.transpose(0, 2, 1)
                return self.inverse_transform_imfs(arr2)
            else:
                raise ValueError("Cannot interpret shape for inverse_transform_imfs")
        else:
            raise ValueError("Unsupported array shape for inverse_transform_imfs")

    def inverse_transform(self, data):
        """
        For backward compatibility: if use_vmd True, tries to inverse as sum of IMFs
        otherwise uses single scaler.
        """
        if self.use_vmd:
            return self.inverse_transform_imfs(data)
        else:
            if self.scaler is None:
                raise RuntimeError("No scaler available for inverse transform")
            return self.scaler.inverse_transform(data)

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler



class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='15min', cols=None,
                 use_vmd=False, num_imfs=5, scaler_type='minmax',
                 vmd_alpha=2000, vmd_tau=0., vmd_DC=0, vmd_init=1, vmd_tol=1e-7,
                 scalers=None):
        """
        Prediction dataset with sliding windows.
        Produces all windows of length seq_len for inference.
        Now supports leakage-free VMD per window when use_vmd=True.
        """

        # scaler_dir = "/content/results/experiment_PatchTST_ftS_sl96_ll48_pl1_dm128_nh16_el3_df256_VMD15_0"
        # num_imfs = 10
        # external_scalers = load_imf_scalers(scaler_dir, num_imfs)

        if size is None:
            self.seq_len = 96   # default sequence length
        else:
            self.seq_len = size[0]

        assert flag in ['pred']
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols

        # vmd/scaler params
        self.use_vmd = use_vmd
        self.num_imfs = int(num_imfs)
        self.scaler_type = scaler_type.lower()
        self.vmd_alpha = vmd_alpha
        self.vmd_tau = vmd_tau
        self.vmd_DC = vmd_DC
        self.vmd_init = vmd_init
        self.vmd_tol = vmd_tol
        self.external_scalers = scalers  # expected dict for vmd, or scaler object for non-vmd

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.cols:
            cols = self.cols.copy()
            if self.target in cols:
                cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        
        # raw ground truth (unscaled scalar series)
        self.data_raw = df_raw[[self.target]].values.astype(float)
        self.df_raw = df_raw

        # time features
        df_stamp = pd.DataFrame()
        df_stamp['date'] = pd.to_datetime(df_raw['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_stamp = data_stamp

        # For non-VMD, precompute scaled data for all windows
        if not self.use_vmd:
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]

            if self.scale:
                if self.external_scalers is not None and isinstance(self.external_scalers, (StandardScaler, MinMaxScaler)):
                    self.scaler = self.external_scalers
                else:
                    self.scaler = MinMaxScaler(feature_range=(0, 1)) if self.scaler_type == 'minmax' else StandardScaler()
                    self.scaler.fit(df_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            self.data_x = data

        # For VMD, IMFs are computed per window in __getitem__ (leakage-free)
        else:
            self.scalers = self.external_scalers  # must be provided (fit on train set)
            # self.data_raw already set above

    def __getitem__(self, index):
        # For non-VMD, just slice precomputed scaled data
        if not self.use_vmd:
            s_begin = index
            s_end = index + self.seq_len

            seq_x = self.data_x[s_begin:s_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]

            if s_end < len(self.data_x):
                seq_y_raw = self.data_raw[s_end]  # raw target scalar
                seq_y_imfs = np.zeros((getattr(self, "num_imfs", 1),), dtype=float)
            else:
                seq_y_raw = np.nan
                seq_y_imfs = np.zeros((getattr(self, "num_imfs", 1),), dtype=float)

            return seq_x, seq_x_mark, seq_y_raw, seq_y_imfs

        # VMD: compute IMFs per window (leakage-free)
        else:
            window = self.data_raw[index : index + self.seq_len].flatten()
            if len(window) < self.seq_len:
                pad = np.zeros(self.seq_len - len(window))
                window = np.concatenate([window, pad])

            # Run VMD on the window only (no future data)
            imfs = _run_vmd(window, alpha=self.vmd_alpha, tau=self.vmd_tau, K=self.num_imfs,
                            DC=self.vmd_DC, init=self.vmd_init, tol=self.vmd_tol)  # shape (K, seq_len)

            # Scale IMFs using external scalers (fit on training set)
            if self.scale:
                imfs_scaled = np.zeros((self.seq_len, self.num_imfs), dtype=float)

                # Initialize dict if missing
                if self.scalers is None:
                    self.scalers = {}

                for i in range(self.num_imfs):
                    if i not in self.scalers:
                        # Create and fit new scaler for this IMF
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        imfs_scaled[:, i] = scaler.fit_transform(imfs[i].reshape(-1, 1)).reshape(-1)
                        self.scalers[i] = scaler
                    else:
                        # Use existing scaler
                        imfs_scaled[:, i] = self.scalers[i].transform(imfs[i].reshape(-1, 1)).reshape(-1)

                # Optional: logging
                if len(self.scalers) == self.num_imfs:
                    print("✅ All IMF scalers fitted and stored.")
            else:
                imfs_scaled = imfs.T  # (seq_len, num_imfs)

            seq_x = imfs_scaled  # shape (seq_len, num_imfs)
            seq_x_mark = self.data_stamp[index : index + self.seq_len]
            seq_y_raw = self.data_raw[index + self.seq_len][0] if (index + self.seq_len) < len(self.data_raw) else np.nan
            seq_y_imfs = np.zeros((self.num_imfs,), dtype=float)  # Not used for prediction

            return seq_x, seq_x_mark, seq_y_raw, seq_y_imfs

    def __len__(self):
        return len(self.data_raw) - self.seq_len + 1

    # ==== Transform helpers ====
    def inverse_transform(self, data):
        if self.use_vmd:
            return self.inverse_transform_imfs(data)
        else:
            if not hasattr(self, "scaler"):
                raise RuntimeError("No scaler available for inverse")
            return self.scaler.inverse_transform(data)

    def inverse_transform_imfs(self, imf_preds):
        """
        Inverse scaling and recombine IMFs to reconstruct original signal.
        imf_preds: shape (seq_len, num_imfs) or (B, seq_len, num_imfs)
        """
        if not self.use_vmd or self.scalers is None:
            return imf_preds.sum(axis=-1)
        K = self.num_imfs
        arr = np.asarray(imf_preds)
        if arr.ndim == 2 and arr.shape[1] == K:  # (seq_len, K)
            out = np.zeros(arr.shape[0])
            for i in range(K):
                sc = self.scalers[i]
                inv_col = sc.inverse_transform(arr[:, i].reshape(-1, 1)).reshape(-1)
                out += inv_col
            return out
        elif arr.ndim == 3:  # (B, seq_len, K)
            B, T, K_ = arr.shape
            out = np.zeros((B, T))
            for b in range(B):
                for i in range(K):
                    sc = self.scalers[i]
                    inv_col = sc.inverse_transform(arr[b, :, i].reshape(-1, 1)).reshape(-1)
                    out[b] += inv_col
            return out
        else:
            raise ValueError(f"Unsupported shape for inverse_transform_imfs: {arr.shape}")
