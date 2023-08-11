import logging
import os
import time

import numpy as np
import torch
from timm.utils import AverageMeter
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from datasets.crowd import Crowd
from datasets.crowd_sh import Crowd_sh
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob
from models.vgg import vgg19
from test import do_test, get_dataloader_by_args
from utils.helper import Save_Handle
from utils.trainer import Trainer


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        # os.environ["WANDB_MODE"] = "offline"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        if args.dataset == 'qnrf':
            self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                      args.crop_size,
                                      args.downsample_ratio,
                                      args.is_gray, x, extra_aug=args.extra_aug
                                      ) for x in ['train', 'val']}
        elif args.dataset in ['sha', 'shb']:
            self.datasets = {x: Crowd_sh(os.path.join(args.data_dir, x),
                                         args.crop_size,
                                         args.downsample_ratio,
                                         args.is_gray, x, extra_aug=args.extra_aug
                                         ) for x in ['train', 'val']}
        else:
            raise NotImplementedError
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                                      if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=1 if x == 'train' else 0,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        self.model = vgg19().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0
        self.wandb_id = None
        if args.resume:
            suf = os.path.splitext(args.resume)[-1]
            if suf == '.tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_mae = checkpoint['best_mae']
                self.best_mse = checkpoint['best_mse']
                self.best_count = checkpoint['best_count']
                if 'wandb_id' in checkpoint:
                    self.wandb_id = checkpoint['wandb_id']
            elif suf == '.pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        self.log_dir = os.path.join(self.save_dir, 'runs')
        # self.writer = SummaryWriter(self.log_dir)
        self.save_list = Save_Handle(max_num=args.max_model_num)
        wandb.init(
            # set the wandb project where this run will be logged
            project="Bayesian-Counting",
            id = self.wandb_id,
            name = os.path.basename(self.args.save_dir),
            # track hyperparameters and run metadata
            config=args,
            resume=True if args.resume else None,
            # sync_tensorboard=True
        )
        self.wandb_id = wandb.run.id

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()
        self.val_epoch()

    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        # Iterate over data.
        for step, (inputs, points, targets, st_sizes) in enumerate(tqdm(self.dataloaders['train'])):
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            outputs = self.model(inputs)
            prob_list = self.post_prob(points, st_sizes)
            loss = self.criterion(prob_list, targets, outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            N = inputs.size(0)
            pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
            res = pre_count - gd_count
            epoch_loss.update(loss.item(), N)
            epoch_mse.update(np.mean(res * res), N)
            epoch_mae.update(np.mean(abs(res)), N)
        wandb.log({
            'train/loss': epoch_loss.avg,
            'train/mae': epoch_mae.avg,
            'train/mse': np.sqrt(epoch_mse.avg),
        }, step=self.epoch)
        # self.writer.add_scalar('train/loss', epoch_loss.avg, self.epoch)
        # self.writer.add_scalar('train/mae', epoch_mae.avg, self.epoch)
        # self.writer.add_scalar('train/mse', np.sqrt(epoch_mse.avg), self.epoch)
        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.avg, np.sqrt(epoch_mse.avg), epoch_mae.avg,
                             time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic,
            'best_mae': self.best_mae,
            'best_mse': self.best_mse,
            'best_count': self.best_count,
            'wandb_id': self.wandb_id
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        # Iterate over data.
        with torch.no_grad():
            for inputs, count, name in tqdm(self.dataloaders['val']):
                inputs = inputs.to(self.device)
                # inputs are images with different sizes
                assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
                outputs = self.model(inputs)
                res = count[0].item() - torch.sum(outputs).item()
                del inputs
                del outputs
                torch.cuda.empty_cache()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time() - epoch_start))
        wandb.log({
            'val/mae': mae,
            'val/mse': mse,
        }, step=self.epoch)
        # self.writer.add_scalar('val/mae', mae, self.epoch)
        # self.writer.add_scalar('val/mse', mse, self.epoch)

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))
    
    def test(self):
        dataloader = get_dataloader_by_args(self.args)
        self.model.load_state_dict(torch.load(os.path.join(self.args.save_dir, 'best_model.pth'), self.device))
        mae, mse = do_test(self.model, self.device, dataloader, self.args.data_dir, self.args.save_dir, locate=True)
        wandb.summary['test_mae'] = mae
        wandb.summary['test_mse'] = mse
