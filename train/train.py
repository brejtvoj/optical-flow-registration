"""
Code is modified from the original RAFT paper.
"""

from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from network import RAFTGMA

from utils_warp import warp
import datasets
from datasets import fetch_dataloader


from torch.cuda.amp import GradScaler

from loss_functions import multi_loss

# exclude extremly large displacements
MAX_FLOW = 250
VAL_FREQ = 500
PLT_FREQ = 100



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                              pct_start=0.0, cycle_momentum=False, anneal_strategy='cos')

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.args = args
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_epe_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}

    def _print_training_status(self):
        metrics_data = [np.mean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data[:-1])).format(*metrics_data[:-1])

        # Compute time left
        time_left_sec = (self.args.num_steps - (self.total_steps+1)) * metrics_data[-1]
        time_left_sec = time_left_sec.astype(np.int)
        time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60, time_left_sec % 3600 % 60)
        time_left_hms = f"{time_left_hms:>12}"
        # print the training status
        print(training_str + metrics_str + time_left_hms)

        # logging running loss to total loss
        self.train_epe_list.append(np.mean(self.running_loss_dict['epe']))
        self.train_steps_list.append(self.total_steps)

        for key in self.running_loss_dict:
            self.running_loss_dict[key] = []

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss_dict:
                self.running_loss_dict[key] = []

            self.running_loss_dict[key].append(metrics[key])

        if self.total_steps % self.args.print_freq == self.args.print_freq-1:
            self._print_training_status()
            self.running_loss_dict = {}

def train(args):
    model = RAFTGMA(args)
    #param_path = "/models/GMA/model_GMA_unsupervised15000.pth"
    #model.load_state_dict(torch.load(param_path), strict=False)
    model.cuda()
    model.train()
    dataset_path = "/datasets/ANHIR_landmarks_half_res/"
    train_loader = fetch_dataloader(args, root=dataset_path)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, args)

    add_noise = False
    should_keep_training = True
    it = 0
    while should_keep_training:
        for i_batch, data_blob in enumerate(train_loader):
            it += 1
            optimizer.zero_grad()
            image1, image2, flow, valid, mask = [x.cuda() for x in data_blob]

            # Data augmentation
            if add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            # Network output
            flow_predictions = model(image1, image2)

            # Losses
            loss = multi_loss(it, image1, image2, flow_predictions, flow, valid, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            if total_steps % PLT_FREQ == PLT_FREQ - 1:
                with torch.no_grad():
                    image_plt = image1[0, :, :, :].detach().unsqueeze(dim=0)
                    flow_plt = flow_predictions[-1][0, :, :, :].detach().unsqueeze(dim=0)
                    image_w = warp(image_plt, flow_plt)
                    image_t = image2[0, :, :, :].detach().unsqueeze(dim=0)
                    print('LOSS: ', loss.item())

                    image_plt = image_plt.squeeze().permute(1, 2, 0).cpu().numpy()
                    image_w = image_w.squeeze().permute(1, 2, 0).cpu().numpy()
                    image_t = image_t.squeeze().permute(1, 2, 0).cpu().numpy()
                    plt.imshow(np.clip(image_plt / 255., 0.0, 1.0))
                    plt.show()
                    plt.imshow(np.clip(image_w / 255., 0.0, 1.0))
                    plt.show()
                    plt.imshow(np.clip(image_t / 255., 0.0, 1.0))
                    plt.show()

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = '/models/GMA/model_GMA_unsupervised_global_%d.pth' % (total_steps+1)
                torch.save(model.state_dict(), PATH)
                model.train()
                if args.stage != 'chairs':
                    model.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'GMA/GMA_model_end.pth'
    torch.save(model.state_dict(), PATH)

    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help="name your experiment")
    parser.add_argument('--stage', default='anhir', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--output', type=str, default='checkpoints', help='output directory to save checkpoints and plots')

    parser.add_argument('--lr', type=float, default=0.0000075)
    parser.add_argument('--num_steps', type=int, default=50000-15000)
    parser.add_argument('--batch_size', type=int, default=1)
    size = 384
    parser.add_argument('--image_size', type=int, nargs='+', default=[size, size])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--upsample-learn', action='store_true', default=False,
                        help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--val_freq', type=int, default=10000,
                        help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='printing frequency')

    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')
    parser.add_argument('--model_name', default='', help='specify model name')

    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    args = parser.parse_args()

    torch.manual_seed(890901234)
    np.random.seed(102491040)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    train(args)
