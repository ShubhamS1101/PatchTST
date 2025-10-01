from PatchTST.data_provider.data_factory import data_provider
from .Exp_basic import Exp_Basic
from PatchTST.utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from PatchTST.utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time
import warnings

from PatchTST.model import Patch

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        """Build the PatchTST model (and wrap for multi-gpu if requested)."""
        model = Patch.Model(self.args).float()
        if getattr(self.args, "use_multi_gpu", False) and getattr(self.args, "use_gpu", False):
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """Return dataset and dataloader for train/val/test/pred"""
        result = data_provider(self.args, flag)
        if len(result) == 3:
            data_set, data_loader, scaler = result
        else:
            data_set, data_loader = result
            scaler = None  # fallback if not returned
        return data_set, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _align_outputs_and_targets(self, outputs, batch_y):
        """
        Align shapes between outputs and batch_y.

        outputs: tensor [B, pred_len, C_out]
        batch_y: tensor [B, pred_len, C_y] (or larger window - caller should slice)
        Return: outputs_sliced, batch_y_sliced (both on same device)
        """
        # ensure both are tensors
        if not torch.is_tensor(outputs):
            outputs = torch.tensor(outputs)

        if not torch.is_tensor(batch_y):
            batch_y = torch.tensor(batch_y)

        # If outputs have a single channel, slice batch_y to single target channel
        if outputs.shape[-1] == 1:
            # keep last channel if multivariate, else keep channel 0
            if batch_y.shape[-1] >= 1:
                target = batch_y[..., -1:] if batch_y.shape[-1] > 1 else batch_y[..., :1]
            else:
                target = batch_y
            return outputs, target.to(outputs.device)

        # If outputs channels match batch_y channels, just return trimmed windows
        if outputs.shape[-1] == batch_y.shape[-1]:
            return outputs, batch_y.to(outputs.device)

        # Otherwise we attempt to use features setting:
        # If multivariate forecasting (MS), keep all channels, else pick last channel (assumed target)
        if getattr(self.args, "features", "S") == "MS":
            # If batch_y has more channels than outputs, try to select the last outputs.shape[-1] channels
            if batch_y.shape[-1] >= outputs.shape[-1]:
                target = batch_y[..., -outputs.shape[-1]:]
                return outputs, target.to(outputs.device)
            else:
                # fallback: expand batch_y by repeating last channel (not ideal, but prevents crash)
                repeat_factor = outputs.shape[-1] // max(1, batch_y.shape[-1])
                target = batch_y.repeat(1, 1, repeat_factor)[:, :, : outputs.shape[-1]]
                return outputs, target.to(outputs.device)
        else:
            # Single-series forecasting (S): pick the last channel of batch_y
            target = batch_y[..., -1:].to(outputs.device) if batch_y.shape[-1] >= 1 else batch_y.to(outputs.device)
            # If outputs expect multiple channels (e.g., num_imfs), but ASWL not used,
            # we leave outputs as-is and compare per-imf (some workflows expect this).
            # But for loss calculation we need shapes equal. If outputs has >1 channel and target has 1,
            # attempt to reduce outputs by taking the last channel or mean across channels.
            if outputs.shape[-1] > 1 and target.shape[-1] == 1:
                # Prefer taking the weighted/mean collapse so that loss can be computed.
                # Use mean across channels as fallback (if ASWL not present).
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

                # forward
                if getattr(self.args, "use_amp", False):
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                # outputs & batch_y alignment
                # keep only last pred_len timesteps for both
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y_window = batch_y[:, -self.args.pred_len:, :]

                outputs_aligned, target_aligned = self._align_outputs_and_targets(outputs, batch_y_window)

                # compute loss on CPU tensors or on device (criterion expects same device)
                loss = criterion(outputs_aligned.to(self.device), target_aligned.to(self.device))
                total_loss.append(loss.item())

        total_loss = np.average(total_loss) if len(total_loss) > 0 else 0.0
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scaler = None
        if getattr(self.args, "use_amp", False):
            scaler = torch.cuda.amp.GradScaler()

        # OneCycleLR is used when args.lradj == 'TST' in original code. Keep behavior.
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

                # forward
                if scaler is not None:
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

                # logging / ETA
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / (iter_count if iter_count > 0 else 1)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # backward
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # lr scheduling
                if getattr(self.args, "lradj", "TST") == 'TST':
                    # original repo used a custom adjust_learning_rate + scheduler.step
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            # epoch end
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

            # alternate lr adjust (if not using TST)
            if getattr(self.args, "lradj", "TST") != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # load best checkpoint (EarlyStopping wrote it to path)
        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            print(f"[Warning] Best model checkpoint not found at {best_model_path}. Skipping load.")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

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

                pred = outputs_np
                true = batch_y_np

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                if i % 20 == 0:
                    input_ = batch_x.detach().cpu().numpy()
                    # guard shapes when visualizing
                    try:
                        gt = np.concatenate((input_[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input_[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    except Exception:
                        # skip visualization on mismatch
                        pass

        if getattr(self.args, "test_flop", False):
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        # reshape to [N, pred_len, C]
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
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

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:, :]
                outputs_np = outputs.detach().cpu().numpy()
                preds.append(outputs_np)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        return
