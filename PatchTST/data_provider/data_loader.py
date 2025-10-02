# updated data_loader.
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
                 use_vmd=False, num_imfs=5, scaler_type='minmax',
                 vmd_alpha=2000, vmd_tau=0., vmd_DC=0, vmd_init=1, vmd_tol=1e-7,
                 scalers=None):
        """
        Extended Dataset_Custom supporting VMD and per-IMF global scalers.

        scalers: optional dict {imf_idx: fitted_scaler} to reuse previously fitted scalers
                 (used for val/test/pred datasets).
        scaler_type: 'minmax' or 'standard'
        vmd_*: hyperparams forwarded to vmdpy.VMD
        """
        # sizes
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # VMD and scaler settings
        self.use_vmd = use_vmd
        self.num_imfs = int(num_imfs) if num_imfs is not None else 5
        self.scaler_type = scaler_type.lower()
        self.vmd_alpha = vmd_alpha
        self.vmd_tau = vmd_tau
        self.vmd_DC = vmd_DC
        self.vmd_init = vmd_init
        self.vmd_tol = vmd_tol

        # Optionally accept precomputed scalers (dict)
        self.external_scalers = scalers

        self.root_path = root_path
        self.data_path = data_path

        # internal holders
        self.scaler = None           # for classic (non-vmd) behavior
        self.scalers = None          # dict for per-IMF scalers when use_vmd=True

        self.__read_data__()

    def __read_data__(self):
        # Read CSV
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # reorder: ['date', other cols..., target]
        cols = list(df_raw.columns)
        if 'date' not in cols or self.target not in cols:
            raise ValueError("data csv must contain 'date' and target column")
        cols_no_date = cols.copy()
        cols_no_date.remove('date')
        if self.target in cols_no_date:
            cols_no_date.remove(self.target)
        df_raw = df_raw[['date'] + cols_no_date + [self.target]]

        # train/val/test split borders (same as before)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Build df_data: either original features/target or IMFs of target if use_vmd
        if self.use_vmd:
            # Decompose the target series into IMFs
            full_signal = df_raw[self.target].values.astype(float)
            u = _run_vmd(full_signal,
                         alpha=self.vmd_alpha, tau=self.vmd_tau,
                         K=self.num_imfs, DC=self.vmd_DC,
                         init=self.vmd_init, tol=self.vmd_tol)  # shape (K, N)

            # Build dataframe of IMFs named imf_0 ... imf_{K-1}
            imf_cols = [f'imf_{i}' for i in range(self.num_imfs)]
            df_imfs = pd.DataFrame(u.T, columns=imf_cols)  # shape (N, K)

            # If features requested were multivariate, currently we only use IMFs
            df_data = df_imfs  # columns are IMFs only

        else:
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]

        # scaling
        if self.scale:
            if self.use_vmd:
                # For VMD: fit / reuse per-IMF scalers on TRAIN only (global per IMF)
                imf_cols = df_data.columns.tolist()  # imf_0, imf_1, ...
                if self.set_type == 0:  # train set: fit scalers (unless external supplied)
                    self.scalers = {}
                    for i, col in enumerate(imf_cols):
                        if self.scaler_type == 'minmax':
                            sc = MinMaxScaler(feature_range=(0, 1))
                        else:
                            sc = StandardScaler()
                        # Fit on training slice of that IMF column
                        train_slice = df_data[col].values[border1s[0]:border2s[0]].reshape(-1, 1)
                        sc.fit(train_slice)
                        self.scalers[i] = sc
                    # Also expose mapping col->scaler name optionally
                else:
                    # val/test: try to use external_scalers passed from train
                    if self.external_scalers is not None:
                        self.scalers = self.external_scalers
                    else:
                        # Fallback: fit scalers on the whole training span (best-effort)
                        self.scalers = {}
                        for i, col in enumerate(imf_cols):
                            if self.scaler_type == 'minmax':
                                sc = MinMaxScaler(feature_range=(0, 1))
                            else:
                                sc = StandardScaler()
                            train_slice = df_data[col].values[border1s[0]:border2s[0]].reshape(-1, 1)
                            sc.fit(train_slice)
                            self.scalers[i] = sc

                # Transform all imf columns using their scaler
                data_scaled = np.zeros_like(df_data.values, dtype=float)
                for i, col in enumerate(imf_cols):
                    data_scaled[:, i] = self.scalers[i].transform(df_data[[col]].values).reshape(-1)
                data = data_scaled
            else:
                # Classic behavior: a single scaler for all features/target
                train_data = df_data[border1s[0]:border2s[0]]
                if self.set_type == 0:
                    # train: fit scaler
                    if self.scaler_type == 'minmax':
                        self.scaler = MinMaxScaler(feature_range=(0, 1))
                    else:
                        self.scaler = StandardScaler()
                    self.scaler.fit(train_data.values)
                else:
                    # val/test: reuse external scaler if provided, else fit on train span
                    if self.external_scalers is not None and isinstance(self.external_scalers, (StandardScaler, MinMaxScaler)):
                        self.scaler = self.external_scalers
                    else:
                        if self.scaler_type == 'minmax':
                            self.scaler = MinMaxScaler(feature_range=(0, 1))
                        else:
                            self.scaler = StandardScaler()
                        self.scaler.fit(train_data.values)

                data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # time encoding (same as your original)
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # Expose scalers: for use by data_provider to pass to val/test/pred
        # If non-vmd, self.scaler is the scalar; if vmd, self.scalers is dict
        # Keep both variables for compatibility
        # self.scalers: dict (imf_idx -> scaler) when use_vmd True
        # self.scaler: single scaler when use_vmd False
        # Note: external_scalers stays untouched

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # same as before
        return len(self.data_x) - self.seq_len - self.pred_len + 1

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
        """
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
        # store raw ground truth
        self.data_raw = df_raw[[self.target]].values
        border1 = 0
        border2 = len(df_raw)

        if self.use_vmd:
            # You must define _run_vmd() somewhere
            full_signal = df_raw[self.target].values.astype(float)
            u = _run_vmd(full_signal,
                         alpha=self.vmd_alpha, tau=self.vmd_tau,
                         K=self.num_imfs, DC=self.vmd_DC,
                         init=self.vmd_init, tol=self.vmd_tol)  # (K, N)
            imf_cols = [f'imf_{i}' for i in range(self.num_imfs)]
            df_imfs = pd.DataFrame(u.T, columns=imf_cols)
            df_data = df_imfs
        else:
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]

        # scaling
        if self.scale:
            if self.use_vmd:
                imf_cols = df_data.columns.tolist()
                if self.external_scalers is None:
                    self.scalers = {}
                    for i, col in enumerate(imf_cols):
                        sc = MinMaxScaler(feature_range=(0, 1)) if self.scaler_type == 'minmax' else StandardScaler()
                        sc.fit(df_data[[col]].values)
                        self.scalers[i] = sc
                else:
                    self.scalers = self.external_scalers

                data_scaled = np.zeros_like(df_data.values, dtype=float)
                for i, col in enumerate(imf_cols):
                    data_scaled[:, i] = self.scalers[i].transform(df_data[[col]].values).reshape(-1)
                data = data_scaled
            else:
                if self.external_scalers is not None and isinstance(self.external_scalers, (StandardScaler, MinMaxScaler)):
                    self.scaler = self.external_scalers
                else:
                    self.scaler = MinMaxScaler(feature_range=(0, 1)) if self.scaler_type == 'minmax' else StandardScaler()
                    self.scaler.fit(df_data.values)
                data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

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

        self.data_x = data
        self.data_stamp = data_stamp
    def __getitem__(self, index):
        s_begin = index
        s_end = index + self.seq_len
    
        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
    
        # single ground truth value (next point)
        if s_end < len(self.data_x):
            seq_y_raw = self.data_raw[s_end]  # next point
        else:
            seq_y_raw = None
    
        return seq_x, seq_x_mark, seq_y_raw



    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

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
        """
        if not self.use_vmd:
            return self.inverse_transform(imf_preds)

        K = self.num_imfs
        arr = np.asarray(imf_preds)

        if arr.ndim == 2 and arr.shape[1] == K:  # (T, K)
            out = np.zeros(arr.shape[0])
            for i in range(K):
                sc = self.scalers[i]
                inv_col = sc.inverse_transform(arr[:, i].reshape(-1, 1)).reshape(-1)
                out += inv_col
            return out

        elif arr.ndim == 3:  # (B, T, K)
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
