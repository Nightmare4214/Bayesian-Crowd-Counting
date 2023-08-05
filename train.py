import argparse
import os

import torch

from utils.regression_trainer import RegTrainer

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--data_dir', default='/home/icml007/Nightmare4214/datasets/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save_dir', default='/home/icml007/Nightmare4214/PyTorch_model/Bayesian',
                        help='directory to save models.')
    parser.add_argument('--dataset', default='qnrf', help='dataset name: qnrf, nwpu, sha, shb')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max_model_num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max_epoch', type=int, default=1000,
                        help='max training epoch')
    parser.add_argument('--val_epoch', type=int, default=5,
                        help='the num of steps to log training information')
    parser.add_argument('--val_start', type=int, default=600,
                        help='the epoch start to val')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='the num of training process')

    parser.add_argument('--is_gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--downsample_ratio', type=int, default=8,
                        help='downsample ratio')

    parser.add_argument('--use_background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background_ratio', type=float, default=1.0,
                        help='background ratio')
    parser.add_argument('--extra_aug', default=False, required=False, action='store_true', help='extra_aug')
    args = parser.parse_args()
    if args.dataset.lower() == 'qnrf':
        args.crop_size = 512
    elif args.dataset.lower() == 'nwpu':
        args.crop_size = 384
        args.val_epoch = 50
    elif args.dataset.lower() == 'sha':
        args.crop_size = 256
    elif args.dataset.lower() == 'shb':
        args.crop_size = 512
    else:
        raise NotImplementedError
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
    trainer.test()
