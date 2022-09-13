# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import shutil
import random
import signal
import subprocess
import sys
import time

from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import model.cifar_resnet as resnet_models
from augmentations import get_aug
from augmentations.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100

parser = argparse.ArgumentParser(description='Clustering-based Mutual Targeting Training')
parser.add_argument('--root', metavar='PATH', type=str,
                    default='',
                    help='path to dataset')
parser.add_argument('--data', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument("--arch", default="resnet18", type=str, help="convnet architecture")
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0050, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='32-32-32', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--checkpoint_freq", type=int, default=50,
                    help="Save the model periodically")


def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()

    # single-node distributed training
    args.rank = 0
    # args.dist_url = 'tcp://localhost:58472'
    args.world_size = args.ngpus_per_node
    # fix_random_seeds(args.seed)
    main_worker(0, args)


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main_worker(gpu, args):
    args.rank += gpu

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = CluMuTModel(args).cuda(gpu)
    model = torch.nn.DataParallel(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    # simulate imbalance data
    transform_train = get_aug(train=True)
    imb_type = 'exp'
    if args.data == 'cifar10':
        dataset = IMBALANCECIFAR10(root='/content/data', imb_type=imb_type, imb_factor=0.1,
                                   rand_number=0, train=True, download=True, transform=transform_train)
    if args.data == 'cifar100':
        dataset = IMBALANCECIFAR100(root='/content/data', imb_type=imb_type, imb_factor=0.1,
                                    rand_number=0, train=True, download=True, transform=transform_train)

    sampler = torch.utils.data.RandomSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.workers,
        pin_memory=True, sampler=sampler
    )

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()

    # create checkpoints file
    model_path = os.path.join(args.checkpoint_dir, "checkpoint.pth")
    dump_checkpoints = os.path.join(args.checkpoint_dir, "checkpoints")
    if not args.rank and not os.path.isdir(dump_checkpoints):
        os.mkdir(dump_checkpoints)

    for epoch in range(start_epoch, args.epochs):
        # sampler.set_epoch(epoch)
        for step, ((xa, xb), _) in enumerate(loader, start=epoch * len(loader)):

            # normalize the prototypes
            with torch.no_grad():
                w = model.module.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                model.module.prototypes.weight.copy_(w)

            xa = xa.cuda(gpu, non_blocking=True)
            xb = xb.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss = model.forward(xa, xb, step)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, model_path)
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    model_path,
                    os.path.join(dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
    if args.rank == 0:
        # save final model
        torch.save(model.module.backbone.state_dict(),
                   args.checkpoint_dir / 'resnet18.pth')


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def mutual_loss(target, h):  # negative cosine similarity

    with torch.no_grad():
        target = target.detach()
        target = F.softmax(target, dim=1)
    loss = F.kl_div(h.softmax(dim=1).log(), target, reduction='sum')

    return loss


class CluMuTModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # build model
        self.backbone = resnet_models.__dict__[args.arch](normalize=True)
        self.out = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # projector        
        sizes = [self.out] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.projector = nn.Sequential(*layers)

        # prototypes
        self.prototypes = nn.Linear(sizes[-2], sizes[-1], bias=False)
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, x_a, x_b, step):
        z_a = self.projector(self.backbone(x_a))
        z_b = self.projector(self.backbone(x_b))

        z_a = nn.functional.normalize(z_a, dim=1, p=2)
        z_b = nn.functional.normalize(z_b, dim=1, p=2)

        # calculate h
        h_a, h_b = self.prototypes(z_a), self.prototypes(z_b)

        # cross-correlation matrix
        c = h_a.T @ h_b
        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)

        # orthogonality regularization
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss_o = on_diag + self.args.lambd * off_diag

        # mutual targeting loss
        loss_mt = mutual_loss(z_a, z_b) / 2 + mutual_loss(z_b, z_a) / 2

        # total loss
        loss = loss_mt + loss_o
        return loss


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


if __name__ == '__main__':
    main()
