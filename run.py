import argparse
import os
import torch

from PatchTST.Exp.Exp_main import Exp_Main

def main():
    parser = argparse.ArgumentParser(description='PatchTST + VMD + AWSL')

    # Data
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of dataset')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='frequency for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # Seq lengths
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

    # Model
    parser.add_argument('--model', type=str, default='PatchTST', help='model name')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size (num channels)')
    parser.add_argument('--e_layers', type=int, default=3, help='number of transformer encoder layers')
    parser.add_argument('--n_heads', type=int, default=16, help='number of attention heads')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--fc_dropout', type=float, default=0.1, help='fc dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride between patches')
    parser.add_argument('--padding_patch', type=str, default='end', help='padding method')

    # Options
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus')
    parser.add_argument('--device_ids', type=str, default='0', help='gpu ids')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer initial learning rate')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--pct_start', type=float, default=0.3, help='OneCycleLR pct_start')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate scheme')

    # Decomposition
    parser.add_argument('--decomposition', action='store_true', help='use series decomposition block')
    parser.add_argument('--kernel_size', type=int, default=25, help='moving average kernel size')

    # VMD + AWSL
    parser.add_argument('--use_vmd', action='store_true', help='use VMD decomposition')
    parser.add_argument('--num_imfs', type=int, default=3, help='number of IMFs from VMD')
    parser.add_argument('--use_aswl', action='store_true', help='use Adaptive Scale-Weighted Layer')

    # Others
    parser.add_argument('--do_train', action='store_true', help='whether to train')
    parser.add_argument('--do_test', action='store_true', help='whether to test')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict')

    args = parser.parse_args()

    # GPU setup
    if args.use_gpu and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
        if args.use_multi_gpu:
            args.device_ids = [int(id_) for id_ in args.device_ids.split(',')]
    else:
        args.device = torch.device('cpu')

    # Experiment name
    setting = '{}_sl{}_pl{}_bs{}_lr{}'.format(
        args.model,
        args.seq_len,
        args.pred_len,
        args.batch_size,
        args.learning_rate
    )

    exp = Exp_Main(args)

    if args.do_train:
        print('>>>>>>> Training : {} >>>>>>>'.format(setting))
        exp.train(setting)

    if args.do_test:
        print('>>>>>>> Testing : {} >>>>>>>'.format(setting))
        exp.test(setting)

    if args.do_predict:
        print('>>>>>>> Predicting : {} >>>>>>>'.format(setting))
        exp.predict(setting)


if __name__ == '__main__':
    main()
