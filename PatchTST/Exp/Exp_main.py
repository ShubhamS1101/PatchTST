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
import logging

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
        """Return dataset, dataloader, and scaler (scaler=None if not provided)."""
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
        train_data, train_loader, _ = self._get_data(flag='train')
        vali_data, vali_loader, _ = self._get_data(flag='val')
        test_data, test_loader, _ = self._get_data(flag='test')

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

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / (iter_count if iter_count > 0 else 1)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
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

        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            print(f"[Warning] Best model checkpoint not found at {best_model_path}. Skipping load.")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader, _ = self._get_data(flag='test')

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

        pred_folder = os.path.join('./results', setting, 'pred/')
        os.makedirs(pred_folder, exist_ok=True)

        if load:
            path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            self.model.load_state_dict(torch.load(path, map_location=self.device))

        self.model.eval()
        _, predict_loader, _ = self._get_data(flag="test")

        preds_all, trues_all = [], []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(predict_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark)

                if self.args.use_vmd:
                    preds_imfs = []
                    for imf_idx in range(outputs.shape[-1]):
                        pred_imf = outputs[..., imf_idx].detach().cpu().numpy()
                        pred_imf = self.scalers[imf_idx].inverse_transform(
                            pred_imf.reshape(-1, 1)
                        ).reshape(pred_imf.shape)
                        preds_imfs.append(pred_imf)

                    preds_denorm = np.stack(preds_imfs, axis=-1)
                    if self.args.use_aswl:
                        w = torch.softmax(self.model.aswl.weights, dim=0).detach().cpu().numpy()
                        preds_final = np.tensordot(preds_denorm, w, axes=([-1], [0]))
                        preds_final = preds_final[..., None]
                    else:
                        preds_final = np.sum(preds_denorm, axis=-1, keepdims=True)
                else:
                    preds_final = self.scaler.inverse_transform(
                        outputs.detach().cpu().numpy().reshape(-1, outputs.shape[-1])
                    ).reshape(outputs.shape)

                preds_all.append(preds_final)
                true_y = batch_y[:, -self.args.pred_len:, -1].detach().cpu().numpy()
                trues_all.append(true_y[..., None])

        preds_all = np.concatenate(preds_all, axis=0)
        trues_all = np.concatenate(trues_all, axis=0)

        np.save(os.path.join(pred_folder, 'real_prediction.npy'), preds_all)
        np.save(os.path.join(pred_folder, 'real_truth.npy'), trues_all)

        logging.info(f"✅ Saved predictions to {pred_folder}")
        return preds_all, trues_all
