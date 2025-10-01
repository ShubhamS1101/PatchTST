#!/usr/bin/env python3
"""
Fixed run.py for PatchTST + VMD + AWSL
Replace your existing run.py with this file.
"""

import argparse
import os
import logging
import sys

import torch

from PatchTST.Exp.Exp_main import Exp_Main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)


def parse_args():
    parser = argparse.ArgumentParser(description='PatchTST + VMD + AWSL')

    # Data / paths
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of dataset')
    parser.add_argument('--data_path', type=str, default='stock.csv', help='data file (inside root_path)')
    parser.add_argument('--features', type=str, default='S', help='forecast task: [M, S, MS]')
    parser.add_argument('--target', type=str, default='close', help='target feature for S or MS')
    parser.add_argument('--freq', type=str, default='h', help='frequency for time features encoding')

    # Checkpoints / saving
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # Sequence lengths
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='label / start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    # Embedding / data encoding
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding; options: [timeF, fixed, learned]')

    parser.add_argument('--num_workers', type=int, default=0, help='num workers for data loading')

    # Model
    parser.add_argument('--model', type=str, default='PatchTST', help='model name')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input channels')
    parser.add_argument('--e_layers', type=int, default=3, help='encoder layers')
    parser.add_argument('--n_heads', type=int, default=16, help='attention heads')
    parser.add_argument('--d_model', type=int, default=128, help='model dimension')
    parser.add_argument('--d_ff', type=int, default=256, help='feed-forward dim')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--fc_dropout', type=float, default=0.1, help='fc dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride between patches')
    parser.add_argument('--padding_patch', type=str, default='end', help='patch padding method')

    # Training / optimization
    parser.add_argument('--device_ids', type=str, default='0', help='comma-separated GPU ids')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use GPU if available')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple GPUs')
    parser.add_argument('--train_epochs', type=int, default=10, help='epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--pct_start', type=float, default=0.3, help='OneCycleLR pct_start')
    parser.add_argument('--lradj', type=str, default='TST', help='lr adjust scheme')

    # Decomposition and smoothing
    parser.add_argument('--decomposition', action='store_true', help='use series decomposition block')
    parser.add_argument('--kernel_size', type=int, default=25, help='moving average kernel size')

    # VMD + AWSL
    parser.add_argument('--use_vmd', action='store_true', help='use VMD decomposition')
    parser.add_argument('--num_imfs', type=int, default=3, help='number of IMFs from VMD')
    # keep the existing flag name but consider standardizing to use_awsl
    parser.add_argument('--use_aswl', action='store_true', help='use Adaptive Scale-Weighted Layer (ASWL/AWSL)')

    # Flow control
    parser.add_argument('--do_train', action='store_true', help='whether to train')
    parser.add_argument('--do_test', action='store_true', help='whether to test')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict')

    # Misc
    parser.add_argument('--itr', type=int, default=1, help='repeats for experiments')
    parser.add_argument('--des', type=str, default='experiment', help='experiment description')

    args = parser.parse_args()
    return args


def validate_args(args):
    # Ensure device_ids is always a string first
    if isinstance(args.device_ids, list):
        device_ids_str = ','.join([str(x) for x in args.device_ids])
    else:
        device_ids_str = args.device_ids

    # GPU availability
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        devices = device_ids_str.replace(' ', '')
        ids = [int(x) for x in devices.split(',') if x != '']
        if len(ids) == 0:
            logging.warning("No device ids parsed from --device_ids; falling back to 0.")
            ids = [0]
        args.device_ids_list = ids    # list for internal use
        args.device_ids = devices     # string for compatibility
        args.gpu = ids[0]
    else:
        # single GPU or CPU
        try:
            args.gpu = int(device_ids_str.split(',')[0].strip())
            args.device_ids = device_ids_str  # keep string
            args.device_ids_list = [args.gpu] # list for internal use
        except Exception:
            args.device_ids = '0'
            args.device_ids_list = [0]
            args.gpu = 0

    # VMD dependency check
    if args.use_vmd:
        try:
            import vmdpy  # noqa: F401
        except Exception as e:
            raise RuntimeError("vmdpy is required for --use_vmd. Install with `pip install vmdpy`") from e

        if args.num_imfs <= 0:
            logging.warning("--num_imfs <= 0; setting num_imfs=1")
            args.num_imfs = 1

    if args.use_aswl and args.num_imfs <= 1:
        logging.warning("ASWL requested but num_imfs <= 1; ASWL will have trivial effect.")

    return args


def build_setting_string(args, iteration=0):
    tags = []
    if args.use_vmd:
        tags.append(f"VMD{args.num_imfs}")
    if args.use_aswl:
        tags.append("ASWL")

    tag_str = "_".join(tags) if tags else "base"

    setting = "{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_{}_{}".format(
        args.des,
        args.model,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_ff,
        tag_str,
        iteration,
    )
    return setting


def main():
    args = parse_args()
    args = validate_args(args)

    logging.info("Arguments:")
    for k, v in sorted(vars(args).items()):
        logging.info("  %s: %s", k, v)

    # run experiments
    exp = Exp_Main(args)

    for ii in range(args.itr):
        setting = build_setting_string(args, ii)
        if args.do_train:
            logging.info(">>>>>>> Training: %s", setting)
            exp.train(setting)

        if args.do_test:
            logging.info(">>>>>>> Testing: %s", setting)
            exp.test(setting)

        if args.do_predict:
            logging.info(">>>>>>> Predicting: %s", setting)
            exp.predict(setting)


if __name__ == '__main__':
    main()
