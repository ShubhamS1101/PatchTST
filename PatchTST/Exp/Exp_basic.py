import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            if self.args.use_multi_gpu:
                # multiple GPUs
                device_ids = [int(id_) for id_ in self.args.device_ids.split(',')]
                os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))
                device = torch.device('cuda:{}'.format(device_ids[0]))
                print(f'Use Multi-GPU: cuda:{device_ids}')
            else:
                # single GPU
                gpu_id = int(self.args.device_ids.split(',')[0])
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                device = torch.device('cuda:0')
                print(f'Use GPU: cuda:{gpu_id}')
        else:
            # fallback to CPU
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
