from PatchTST.data_provider.data_factory import data_provider
from PatchTST.model.trend_loss import trend_loss
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
    def init(self, args):
        super(Exp_Main, self).init(args)
        self.scaler = None
        self.scalers = None
        self.model = self._build_model().to(self.device) # ensure model built on init     # dict/list of per-imf scalers (vmd)
    
    def _build_model(self):
        model = Patch.Model(self.args).float()
        if getattr(self.args, "use_multi_gpu", False) and getattr(self.args, "use_gpu", False):
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        result = data_provider(self.args, flag)
        if len(result) == 3:
            data_set, data_loader, scaler = result
        else:
            data_set, data_loader = result
            scaler = None
        return data_set, data_loader, scaler

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return trend_loss

    # === added/resume ===
    def _resume_checkpoint(self, setting):
        """Load checkpoint if available (prefers full, else weights-only)."""
        path_dir = os.path.join(self.args.checkpoints, setting)
        path_dir = '/content/PatchTST/PatchTST/checkpoints/experiment_PatchTST_ftS_sl96_ll96_pl1_dm256_nh16_el5_df256_VMD8_ASWL_0/'
        full_path = os.path.join(path_dir, "checkpoint_full.pth")
        best_path = os.path.join(path_dir, "checkpoint.pth")

        print(best_path)

        if os.path.exists(full_path):
            print(f"🔄 Resuming full checkpoint from {full_path}")
            checkpoint = torch.load(full_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self._loaded_optimizer_state = checkpoint.get("optimizer_state_dict", None)
            print("✅ Model + optimizer state loaded successfully.")
        elif os.path.exists(best_path):
            print(f"🔄 Resuming from best weights (no optimizer): {best_path}")
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            self._loaded_optimizer_state = None
        else:
            print("[Info] No checkpoint found to resume from.")
            self._loaded_optimizer_state = None

    def _align_outputs_and_targets(self, outputs, batch_y):
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
                if outputs_aligned.shape[1] > 1:
                    pred = outputs_aligned[:, 1:, :]
                    true = target_aligned[:, 1:, :]
                    prev_pred = outputs_aligned[:, :-1, :]
                    prev_true = target_aligned[:, :-1, :]
                    loss = criterion(pred, true, prev_pred, prev_true, alpha=0.2)
                else:
                    prev_true = batch_x[:, -1:, :]
                    prev_pred = outputs_aligned.detach()
                    loss = criterion(outputs_aligned, target_aligned, prev_pred, prev_true, alpha=0.2)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss) if len(total_loss) > 0 else 0.0
        self.model.train()
        return total_loss

    def train(self, setting, resume=False):
        train_data, train_loader, train_scaler = self._get_data(flag="train")
        vali_data, vali_loader, _ = self._get_data(flag="val")
        test_data, test_loader, _ = self._get_data(flag="test")

        if getattr(self.args, "use_vmd", False):
            self.scalers = train_scaler
        else:
            self.scaler = train_scaler

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        # === resume checkpoint logic ===
        if resume:
            self._resume_checkpoint(setting)

        model_optim = self._select_optimizer()
        if getattr(self, "_loaded_optimizer_state", None) is not None:
            try:
                model_optim.load_state_dict(self._loaded_optimizer_state)
                print("✅ Optimizer state resumed.")
            except Exception as e:
                print(f"[Warning] Could not load optimizer state: {e}")

        criterion = self._select_criterion()
        amp_scaler = torch.cuda.amp.GradScaler() if getattr(self.args, "use_amp", False) else None

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=max(1, len(train_loader)),
            pct_start=getattr(self.args, "pct_start", 0.3),
            epochs=max(1, self.args.train_epochs),
            max_lr=self.args.learning_rate,
        )

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        print(f"🚀 Starting training for {self.args.train_epochs} epochs...")

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

                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y_window = batch_y[:, -self.args.pred_len:, :]
                outputs_aligned, target_aligned = self._align_outputs_and_targets(outputs, batch_y_window)

                if outputs_aligned.shape[1] > 1:
                    pred = outputs_aligned[:, 1:, :]
                    true = target_aligned[:, 1:, :]
                    prev_pred = outputs_aligned[:, :-1, :]
                    prev_true = target_aligned[:, :-1, :]
                    loss = criterion(pred, true, prev_pred, prev_true, alpha=0.2)
                else:
                    prev_true = batch_x[:, -1:, :]
                    prev_pred = outputs_aligned.detach()
                    loss = criterion(outputs_aligned, target_aligned, prev_pred, prev_true, alpha=0.2)

                train_loss.append(loss.item())

                if amp_scaler is not None:
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(model_optim)
                    amp_scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if getattr(self.args, "lradj", "TST") == "TST":
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss_avg = np.average(train_loss) if len(train_loss) > 0 else 0.0
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print(f"Epoch: {epoch+1} | Train: {train_loss_avg:.6f}, Val: {vali_loss:.6f}, Test: {test_loss:.6f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            if getattr(self.args, "lradj", "TST") != "TST":
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print(f"Updating learning rate to {scheduler.get_last_lr()[0]:.6f}")

            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": model_optim.state_dict(),
            }, os.path.join(path, "checkpoint_full.pth"))

        best_model_path = os.path.join(path, "checkpoint.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print(f"✅ Loaded best model from {best_model_path}")
        else:
            print("[Warning] Best model checkpoint not found.")

        save_dir = os.path.join("./results", setting)
        os.makedirs(save_dir, exist_ok=True)

        try:
            if getattr(self.args, "use_vmd", False) and (self.scalers is not None):
                for i, sc in enumerate(self.scalers):
                    joblib.dump(sc, os.path.join(save_dir, f"scaler_imf{i}.pkl"))
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
