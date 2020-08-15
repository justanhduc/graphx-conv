import argparse

parser = argparse.ArgumentParser('GraphX-convolution')
parser.add_argument('config_file', type=str, help='config file to dictate training/testing')
parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu number')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import neuralnet_pytorch.gin_nnt as gin

from networks import *
from data_loader import ShapeNet, collate

config_file = args.config_file
backup_files = ['train.py', 'networks.py', 'data_loader.py', 'ops.py'] + [config_file]

gin.external_configurable(CNN18Encoder, 'cnn18_enc')
gin.external_configurable(PointCloudEncoder, 'pc_enc')
gin.external_configurable(PointCloudDecoder, 'pc_dec')
gin.external_configurable(PointCloudResDecoder, 'pc_resdec')
gin.external_configurable(PointCloudResGraphXUpDecoder, 'pc_upresgraphxdec')
gin.external_configurable(PointCloudResLowRankGraphXUpDecoder, 'pc_upreslowrankgraphxdec')


@gin.configurable('GraphX')
def train_valid(data_root, name, img_enc, pc_enc, pc_dec, optimizer, scheduler, adain=True, projection=True,
                decimation=None, color_img=False, n_points=250, bs=4, lr=5e-5, weight_decay=1e-5, gamma=.3,
                milestones=(5, 8), n_epochs=10, print_freq=1000, val_freq=10000, checkpoint_folder=None):
    if decimation is not None:
        pc_dec = partial(pc_dec, decimation=decimation)

    net = PointcloudDeformNet((bs,) + (3 if color_img else 1, 224, 224), (bs, n_points, 3), img_enc, pc_enc, pc_dec,
                              adain=adain, projection=projection, weight_decay=None)
    print(net)
    solver = T.optim.Adam(net.trainable, 1e-4, weight_decay=0) if optimizer is None \
        else optimizer(net.trainable, lr, weight_decay=weight_decay)
    scheduler = scheduler(solver, milestones=milestones, gamma=gamma) if scheduler is not None else None

    train_data = ShapeNet(path=data_root, grayscale=not color_img, type='train', n_points=n_points)
    sampler = WeightedRandomSampler(train_data.sample_weights, len(train_data), True)
    train_loader = DataLoader(train_data, batch_size=bs, num_workers=1, collate_fn=collate, drop_last=True,
                              sampler=sampler)

    val_data = ShapeNet(path=data_root, grayscale=not color_img, type='valid', num_vals=10 * len(os.listdir(data_root)),
                        n_points=n_points)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=1, collate_fn=collate, drop_last=True)

    mon.model_name = name
    mon.print_freq = print_freq
    mon.num_iters = len(train_data) // bs
    mon.set_path(checkpoint_folder)
    if checkpoint_folder is None:
        mon.backup(backup_files)
        mon.dump_rep('network', net)
        mon.dump_rep('optimizer', solver)
        if scheduler is not None:
            mon.dump_rep('scheduler', scheduler)

        def save_checkpoint():
            states = {
                'states': mon.epoch,
                'model_state_dict': net.state_dict(),
                'opt_state_dict': solver.state_dict()
            }
            if scheduler is not None:
                states['scheduler_state_dict'] = scheduler.state_dict()

            mon.dump(name='training.pt', obj=states, method='torch', keep=5)

        mon.schedule(save_checkpoint, when=mon._end_epoch_)
        print('Training...')
    else:
        states = mon.load('training.pt', type='torch')
        mon.epoch = states['epoch']
        net.load_state_dict(states['model_state_dict'])
        net.optim['optimizer'].load_state_dict(states['opt_state_dict'])
        if net.optim['scheduler']:
            net.optim['scheduler'].load_state_dict(states['scheduler_state_dict'])

        print('Resume from epoch %d...' % mon.epoch)

    mon.run_training(net, solver, train_loader, n_epochs, scheduler=scheduler, eval_loader=val_loader,
                     valid_freq=val_freq, reduce='mean')
    print('Training finished!')


if __name__ == '__main__':
    gin.parse_config_file(config_file)
    train_valid()
