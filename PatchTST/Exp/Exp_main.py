from PatchTST.data_provider.data_factory import data_provider
from .Exp_basic import Exp_Basic
from PatchTST.utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from PatchTST.utils.metrics import metric
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time
import warnings
import logging

from PatchTST.model import Patch

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        # placeholders for scalers that will be set during train or loaded in predict
        self.scaler = None       # single scaler (non-vmd)
        self.scalers = None      # dict/list of per-imf scalers (vmd)
    
    def _build_model(self):
        """Build the PatchTST model (and wrap for multi-gpu if requested)."""
        model = Patch.Model(self.args).float()
        if getattr(self.args, "use_multi_gpu", False) and getattr(self.args, "use_gpu", False):
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """
        Return dataset, dataloader, scaler.
        scaler will be None if data_provider didn't return one.
        """
        result = data_provider(self.args, flag)
        if len(result) == 3:
            data_set, data_loader, scaler = result
        else:
            data_set, data_loader = result
            scaler = None
        return data_set, data_loader, scaler

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _align_outputs_and_targets(self, outputs, batch_y):
        """
        Align shapes between outputs and batch_y.
        """
        if not torch.is_tensor(outputs):
            outputs = torch.tensor(outputs)
        if not torch.is_tensor(batch_y):
            batch_y = torch.tensor(batch_y)

        if outputs.shape[-1] == 1:
            if batch_y.shape[-1] >= 1:
                target = batch_y[..., -1:] if batch_y.shape[-1] > 1 else batch_y[..., :1]
            else:
                target = batch_y
            return outputs, target.to(outputs.device)

        if outputs.shape[-1] == batch_y.shape[-1]:
            return outputs, batch_y.to(outputs.device)

        if getattr(self.args, "features", "S") == "MS":
            if batch_y.shape[-1] >= outputs.shape[-1]:
                target = batch_y[..., -outputs.shape[-1]:]
                return outputs, target.to(outputs.device)
            else:
                repeat_factor = outputs.shape[-1] // max(1, batch_y.shape[-1])
                target = batch_y.repeat(1, 1, repeat_factor)[:, :, : outputs.shape[-1]]
                return outputs, target.to(outputs.device)
        else:
            target = batch_y[..., -1:].to(outputs.device) if batch_y.shape[-1] >= 1 else batch_y.to(outputs.device)
            if outputs.shape[-1] > 1 and target.shape[-1] == 1:
                outputs_collapsed = outputs.mean(dim=-1, keepdim=True)
                return outputs_collapsed, target
            return outputs, target

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if getattr(self.args, "use_amp", False):
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y_window = batch_y[:, -self.args.pred_len:, :]

                outputs_aligned, target_aligned = self._align_outputs_and_targets(outputs, batch_y_window)
                loss = criterion(outputs_aligned.to(self.device), target_aligned.to(self.device))
                total_loss.append(loss.item())

        total_loss = np.average(total_loss) if len(total_loss) > 0 else 0.0
        self.model.train()
        return total_loss

    def train(self, setting):
        # get data and scalers for train/val/test
        train_data, train_loader, train_scaler = self._get_data(flag='train')
        vali_data, vali_loader, _ = self._get_data(flag='val')
        test_data, test_loader, _ = self._get_data(flag='test')

        # store scalers from training dataset for later inverse transform/save
        if getattr(self.args, "use_vmd", False):
            # train_scaler expected to be a dict/list of per-imf scalers
            self.scalers = train_scaler
        else:
            self.scaler = train_scaler

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        amp_scaler = None
        if getattr(self.args, "use_amp", False):
            amp_scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=max(1, train_steps),
            pct_start=getattr(self.args, "pct_start", 0.3),
            epochs=max(1, self.args.train_epochs),
            max_lr=self.args.learning_rate
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if amp_scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        outputs = outputs[:, -self.args.pred_len:, :]
                        batch_y_window = batch_y[:, -self.args.pred_len:, :]
                        outputs_aligned, target_aligned = self._align_outputs_and_targets(outputs, batch_y_window)
                        loss = criterion(outputs_aligned.to(self.device), target_aligned.to(self.device))
                else:
                    outputs = self.model(batch_x)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y_window = batch_y[:, -self.args.pred_len:, :]
                    outputs_aligned, target_aligned = self._align_outputs_and_targets(outputs, batch_y_window)
                    loss = criterion(outputs_aligned.to(self.device), target_aligned.to(self.device))

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / (iter_count if iter_count > 0 else 1)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if amp_scaler is not None:
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(model_optim)
                    amp_scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if getattr(self.args, "lradj", "TST") == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss_avg = np.average(train_loss) if len(train_loss) > 0 else 0.0
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss_avg, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if getattr(self.args, "lradj", "TST") != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # load best checkpoint (EarlyStopping wrote it to path)
        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        else:
            print(f"[Warning] Best model checkpoint not found at {best_model_path}. Skipping load.")

        # ------------------ SAVE SCALERS ------------------
        # Save scalers so predict() can load them later. Save under ./results/<setting>/
        save_dir = os.path.join('./results', setting)
        os.makedirs(save_dir, exist_ok=True)

        try:
            if getattr(self.args, "use_vmd", False) and (self.scalers is not None):
                # assume self.scalers is a dict or list-like of fitted scaler objects
                for i, sc in enumerate(self.scalers):
                    joblib.dump(sc, os.path.join(save_dir, f"scaler_imf{i}.pkl"))
                for i in range(len(self.scalers)):
                    joblib.dump(self.scalers[i], os.path.join(save_dir, f"hello{i}.pkl"))
                    print(type(self.scalers[i]))
                print(f"✅ Saved {len(self.scalers)} IMF scalers to {save_dir}")
            elif self.scaler is not None:
                joblib.dump(self.scaler, os.path.join(save_dir, "scaler.pkl"))
                print(f"✅ Saved single scaler to {save_dir}")
            else:
                print("[Info] No scaler object found to save.")
        except Exception as e:
            print(f"[Warning] Failed to save scalers: {e}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader, _ = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'),
                                                  map_location=self.device))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y_window = batch_y[:, -self.args.pred_len:, :]

                outputs_aligned, target_aligned = self._align_outputs_and_targets(outputs, batch_y_window)

                outputs_np = outputs_aligned.detach().cpu().numpy()
                batch_y_np = target_aligned.detach().cpu().numpy()

                preds.append(outputs_np)
                trues.append(batch_y_np)
                inputx.append(batch_x.detach().cpu().numpy())

                if i % 20 == 0:
                    try:
                        gt = np.concatenate((inputx[-1][0, :, -1], batch_y_np[0, :, -1]), axis=0)
                        pd = np.concatenate((inputx[-1][0, :, -1], outputs_np[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    except Exception:
                        pass

        if getattr(self.args, "test_flop", False):
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds = np.array(preds).reshape(-1, preds[0].shape[-2], preds[0].shape[-1])
        trues = np.array(trues).reshape(-1, trues[0].shape[-2], trues[0].shape[-1])
        inputx = np.array(inputx).reshape(-1, inputx[0].shape[-2], inputx[0].shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        with open("result.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f.write('\n\n')

        np.save(folder_path + 'pred.npy', preds)
        return

    def predict(self, setting, load=True):
        import numpy as np
        import os
        import logging
        import torch

        pred_folder = os.path.join('./results', setting, 'pred/')
        os.makedirs(pred_folder, exist_ok=True)

        # ---- Load trained model ----
        if load:
            path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            self.model.load_state_dict(torch.load(path, map_location=self.device))

        # ---- Prediction data loader ----
        predict_data, predict_loader, _ = self._get_data(flag="pred")

        # ---- Use scalers from the dataset ----
        if hasattr(predict_data, "scalers"):
            self.scalers = predict_data.scalers
        if hasattr(predict_data, "scaler"):
            self.scaler = predict_data.scaler

        preds_final_all = []   # denormalized + summed predictions
        trues_final_all = []   # raw ground truth
        preds_imfs_all = []    # normalized IMF predictions
        trues_imfs_all = []    # normalized IMF ground truth

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_mark, batch_y_raw, batch_y_imfs) in enumerate(predict_loader):

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # ---- Forward pass ----
                outputs = self.model(batch_x, batch_x_mark)   # [B, pred_len, K]
                outputs = outputs[:, -self.args.pred_len:, :] # keep last pred_len

                # ---- Store normalized IMF predictions ----
                preds_imfs_all.append(outputs[:, -1, :].detach().cpu().numpy())  # (B, K)

                # ---- Inverse transform for reconstruction ----
                if getattr(self.args, "use_vmd", False):
                    outputs_denorm = predict_data.inverse_transform(outputs.detach().cpu().numpy())
                else:
                    out_np = outputs.detach().cpu().numpy().reshape(-1, outputs.shape[-1])
                    if hasattr(self, "scaler") and self.scaler is not None:
                        inv = self.scaler.inverse_transform(out_np)
                        outputs_denorm = inv.reshape(outputs.shape)
                    else:
                        outputs_denorm = outputs.detach().cpu().numpy()

                # ---- Collapse to final scalar prediction ----
                if outputs_denorm.ndim == 3:   # (B, T, K)
                    preds_single = outputs_denorm[:, -1, :].sum(axis=-1)  # (B,)
                elif outputs_denorm.ndim == 2: # (B, K)
                    preds_single = outputs_denorm.sum(axis=-1)            # (B,)
                else:
                    raise ValueError(f"Unexpected output shape: {outputs_denorm.shape}")

                preds_final_all.append(preds_single)

                # ---- Collect ground truth ----
                if batch_y_raw is not None:
                    trues_final_all.append(batch_y_raw.reshape(-1))   # raw scalar

                if batch_y_imfs is not None:
                    trues_imfs_all.append(batch_y_imfs.reshape(-1, batch_y_imfs.shape[-1]))  # (B, K)

        # ---- Concatenate everything ----
        preds_final_all = np.concatenate(preds_final_all, axis=0)
        trues_final_all = np.concatenate(trues_final_all, axis=0) if trues_final_all else np.array([])
        preds_imfs_all = np.concatenate(preds_imfs_all, axis=0) if preds_imfs_all else np.array([])
        trues_imfs_all = np.concatenate(trues_imfs_all, axis=0) if trues_imfs_all else np.array([])

        print(f"[DEBUG] pred_final: {preds_final_all.shape}, true_final: {trues_final_all.shape}, "
              f"pred_imfs: {preds_imfs_all.shape}, true_imfs: {trues_imfs_all.shape}")

        # ---- Save outputs ----
        np.save(os.path.join(pred_folder, 'pred_final.npy'), preds_final_all)
        np.save(os.path.join(pred_folder, 'true_final.npy'), trues_final_all)
        np.save(os.path.join(pred_folder, 'pred_imfs.npy'), preds_imfs_all)
        np.save(os.path.join(pred_folder, 'true_imfs.npy'), trues_imfs_all)

        logging.info(f"✅ Saved predictions to {pred_folder}")

        return preds_final_all, trues_final_all, preds_imfs_all, trues_imfs_all
