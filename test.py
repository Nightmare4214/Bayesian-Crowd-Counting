import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from datasets.crowd import Crowd
from datasets.crowd_sh import Crowd_sh
from models.vgg import vgg19

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data_dir', default='/home/icml007/Nightmare4214/datasets/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save_dir', default='/home/icml007/Nightmare4214/PyTorch_model/bayesian',
                        help='model directory')
    parser.add_argument('--dataset', default='qnrf', help='dataset name: qnrf, nwpu, sha, shb')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='the crop size of the train image')
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


def get_dataloader_by_args(args):
    if args.dataset.lower() == 'qnrf':
        datasets = Crowd(os.path.join(args.data_dir, 'test'), args.crop_size, 8, is_gray=False, method='val')
    elif args.dataset.lower() in ['sha', 'shb']:
        datasets = Crowd_sh(os.path.join(args.data_dir, 'test'), args.crop_size, 8, is_gray=False, method='val')
    else:
        raise NotImplementedError
    return torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                       num_workers=0, pin_memory=False)


def do_test(model, device, dataloader, data_dir, save_dir, locate=True, **kwargs):
    model.eval()
    epoch_minus = []
    if os.path.isdir(save_dir):
        model_dir = save_dir
    else:
        model_dir = os.path.dirname(save_dir)
    if locate:
        locate_dir = os.path.join(model_dir, 'predict')
        os.makedirs(locate_dir, exist_ok=True)

    with torch.no_grad():
        for inputs, count, name in tqdm(dataloader):
            inputs = inputs.to(device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1'
            outputs = model(inputs)
            temp_minu = count[0].item() - torch.sum(outputs).item()
            # print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
            del inputs
            del outputs
            torch.cuda.empty_cache()
            epoch_minus.append(temp_minu)

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)
    with open(os.path.join(model_dir, 'predict.log'), 'w') as f:
        f.write(log_str + '\n')
    return mae, mse

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    dataloader = get_dataloader_by_args(args)
    model = vgg19()
    device = torch.device('cuda')
    model = model.to(device)
    model_path = args.save_dir
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, 'best_model.pth')
    model.load_state_dict(torch.load(model_path, device))
    do_test(model, device, dataloader, args.data_dir, args.save_dir, locate=True)
    
