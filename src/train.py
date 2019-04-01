import argparse

parser = argparse.ArgumentParser('GraphX-convolution')
parser.add_argument('path', type=str, help='path to the ShapeNet train/test spit')
parser.add_argument('--no_graphx', action='store_true', help='not use graphx')
parser.add_argument('-n', '--n_points', type=int, default=2000, help='initial point cloud size')
parser.add_argument('-c', '--color', action='store_true', help='if specified, color image is used instead of gray')
parser.add_argument('-b', '--batchsize', type=int, default=4, help='batch size')
parser.add_argument('-l', '--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('-w', '--wd', type=float, default=1e-5, help='weight decay')
parser.add_argument('--no_schedule', action='store_false', help='not use lr schedule')
parser.add_argument('--gamma', default=.3, type=float, help='multiplier for lr')
parser.add_argument('--milestones', default=(5, 8), type=int, nargs='+', help='epoch to decrease lr')
parser.add_argument('-e', '--n_epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--print_freq', default=1000, type=int, help='statistics displaying frequency')
parser.add_argument('--valid_freq', default=0, type=int, help='validation frequency. default is no validation')
parser.add_argument('--save_results', action='store_true', help='save validation visualization results')
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint to resume training')
parser.add_argument('-v', '--version', type=int, default=-1, help='version of the checkpoint')
parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu number')
args = parser.parse_args()

import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch as T
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim.lr_scheduler import MultiStepLR

from networks import Pixel2Pointcloud
from data_loader import ShapeNet, collate
from monitor import Monitor
from utils import *

path = args.path
file_list = os.listdir(path)
color_img = args.color
shape = (3 if color_img else 1, 224, 224)
n_points = args.n_points
bs = args.batchsize
lr = args.lr
weight_decay = args.wd
gamma = args.gamma
milestones = args.milestones
n_epochs = args.n_epochs
print_freq = args.print_freq
val_freq = args.valid_freq
save_results = False

# resume training
checkpoint_folder = args.checkpoint
version = args.version


def train_valid():
    net = Pixel2Pointcloud(3 if color_img else 1, n_points,
                           optimizer=lambda x: T.optim.Adam(x, lr, weight_decay=weight_decay),
                           scheduler=lambda x: MultiStepLR(x, milestones=milestones, gamma=gamma),
                           use_graphx=not args.no_graphx)
    print(net)

    train_data = ShapeNet(file_list, path, grayscale=not color_img, type='train', n_points=n_points)
    sampler = WeightedRandomSampler(train_data.sample_weights, len(train_data), True)
    train_loader = DataLoader(train_data, batch_size=bs, num_workers=10, collate_fn=collate, drop_last=True,
                              sampler=sampler)

    if val_freq:
        val_data = ShapeNet(file_list, args.path, grayscale=not color_img, type='valid', num_vals=10 * len(file_list),
                            n_points=n_points)
        val_loader = DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=10, collate_fn=collate,
                                drop_last=True)

    mon = Monitor('graphx-conv', print_freq=print_freq, num_iters=len(train_data) // bs,
                  use_tensorboard=True)
    mon.dump_rep('network', net)
    mon.dump_rep('optimizer', net.optimizer)
    if net.scheduler:
        mon.dump_rep('scheduler', net.scheduler)

    states = {
        'model_state_dict': net.state_dict(),
        'opt_state_dict': net.optimizer.state_dict()
    }
    if net.scheduler:
        states['scheduler_state_dict'] = net.scheduler.state_dict()

    print('Training...')
    for epoch in range(n_epochs):
        states['epoch'] = epoch
        mon.dump('training.pt', states, 'torch', 5)

        if net.scheduler:
            net.scheduler.step(epoch=epoch)
            mon.plot('lr', net.scheduler.optimizer.param_groups[0]['lr'])

        for it, batch in enumerate(train_loader):
            with mon:
                init_pc, image, gt_pc = batch
                if T.cuda.is_available():
                    init_pc = init_pc.cuda()
                    image = image.cuda()
                    gt_pc = [pc.cuda() for pc in gt_pc] if isinstance(gt_pc, (list, tuple)) else gt_pc.cuda()

                loss = net.learn(image, init_pc, gt_pc)
                if np.isnan(loss) or np.isinf(loss):
                    raise ValueError('NaN loss. Training failed!')

                mon.plot('chamfer', loss)

                if val_freq:
                    if it % val_freq == 0:
                        net.eval()
                        with T.set_grad_enabled(False):
                            for itt, batch in enumerate(val_loader):
                                init_pc, img, gt_pc = batch
                                if T.cuda.is_available():
                                    init_pc = init_pc.cuda()
                                    img = img.cuda()
                                    gt_pc = [pc.cuda() for pc in gt_pc] if isinstance(gt_pc,
                                                                                      (list, tuple)) else gt_pc.cuda()

                                loss_dict, pred_pc = net.loss(img, init_pc, gt_pc, reduce='mean')
                                pred_pc = to_numpy(pred_pc)
                                loss = to_numpy(loss_dict['chamfer'])
                                mon.plot('valid chamfer', loss)

                                if save_results:
                                    mon.scatter('valid pred pointcloud %d' % itt, pred_pc)
                                    for ii in range(pred_pc.shape[0]):
                                        mon.dump('valid pred pointcloud %d %d.xyz' % (ii, itt), pred_pc[ii], 'txt',
                                                 delimiter=' ')

    mon.dump('training.pt', states, 'torch')
    print('Training finished!')


def resume():
    train_data = ShapeNet(file_list, args.path, grayscale=not color_img, type='train', n_points=n_points)
    sampler = WeightedRandomSampler(train_data.sample_weights, len(train_data), True)
    train_loader = DataLoader(train_data, batch_size=bs, num_workers=10, collate_fn=collate, drop_last=True,
                              sampler=sampler)

    if val_freq:
        val_data = ShapeNet(file_list, args.path, grayscale=not color_img, type='valid', num_vals=10 * len(file_list),
                            n_points=n_points)
        val_loader = DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=10, collate_fn=collate,
                                drop_last=True)

    mon = Monitor(current_folder=checkpoint_folder, print_freq=print_freq, num_iters=len(train_data) // bs,
                  use_tensorboard=True)
    states = mon.load('training.pt', type='torch')
    mon.set_iter(states['epoch'] * len(train_data) // bs)

    net = Pixel2Pointcloud(3 if color_img else 1, n_points,
                           optimizer=lambda x: T.optim.Adam(x, lr, weight_decay=weight_decay),
                           scheduler=lambda x: MultiStepLR(x, milestones=milestones, gamma=gamma),
                           use_graphx=not args.no_graphx)
    print(net)

    net.load_state_dict(states['model_state_dict'])
    net.optimizer.load_state_dict(states['opt_state_dict'])
    if net.scheduler:
        net.scheduler.load_state_dict(states['scheduler_state_dict'])

    print('Resume from epoch %d...' % states['epoch'])
    for epoch in range(states['epoch'], n_epochs):
        states['epoch'] = epoch
        mon.dump('training.pt', states, 'torch', 5)

        if net.scheduler:
            net.scheduler.step(epoch=epoch)
            mon.plot('lr', net.scheduler.optimizer.param_groups[0]['lr'])

        for it, batch in enumerate(train_loader):
            with mon:
                init_pc, image, gt_pc = batch
                if T.cuda.is_available():
                    init_pc = init_pc.cuda()
                    image = image.cuda()
                    gt_pc = [pc.cuda() for pc in gt_pc] if isinstance(gt_pc, (list, tuple)) else gt_pc.cuda()

                loss = net.learn(image, init_pc, gt_pc)
                if np.isnan(loss) or np.isinf(loss):
                    raise ValueError('NaN loss. Training failed!')

                mon.plot('chamfer', loss)

                if val_freq:
                    if it % val_freq == 0:
                        net.eval()
                        with T.set_grad_enabled(False):
                            for itt, batch in enumerate(val_loader):
                                init_pc, img, gt_pc = batch
                                if T.cuda.is_available():
                                    init_pc = init_pc.cuda()
                                    img = img.cuda()
                                    gt_pc = [pc.cuda() for pc in gt_pc] if isinstance(gt_pc,
                                                                                      (list, tuple)) else gt_pc.cuda()

                                loss_dict, pred_pc = net.loss(img, init_pc, gt_pc, 'mean')
                                pred_pc = to_numpy(pred_pc)
                                loss = to_numpy(loss_dict['chamfer'])
                                mon.plot('valid chamfer', loss)

                                if save_results:
                                    mon.scatter('valid pred pointcloud %d' % itt, pred_pc)
                                    for ii in range(pred_pc.shape[0]):
                                        mon.dump('valid pred pointcloud %d %d.xyz' % (ii, itt), pred_pc[ii], 'txt',
                                                 delimiter=' ')

    mon.dump('training.pt', states, 'torch')
    print('Training finished!')


if __name__ == '__main__':
    train_valid() if checkpoint_folder is None else resume()
