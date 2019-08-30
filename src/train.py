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
                              adain=adain, projection=projection,
                              optimizer=lambda x: optimizer(x, lr, weight_decay=weight_decay),
                              scheduler=lambda x: scheduler(x, milestones=milestones, gamma=gamma),
                              weight_decay=None)
    print(net)

    train_data = ShapeNet(path=data_root, grayscale=not color_img, type='train', n_points=n_points)
    sampler = WeightedRandomSampler(train_data.sample_weights, len(train_data), True)
    train_loader = DataLoader(train_data, batch_size=bs, num_workers=1, collate_fn=collate, drop_last=True,
                              sampler=sampler)

    val_data = ShapeNet(path=data_root, grayscale=not color_img, type='valid', num_vals=10 * len(os.listdir(data_root)),
                        n_points=n_points)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=1, collate_fn=collate, drop_last=True)

    if checkpoint_folder is None:
        mon = nnt.Monitor(name, print_freq=print_freq, num_iters=len(train_data) // bs, use_tensorboard=True)
        mon.copy_files(backup_files)

        mon.dump_rep('network', net)
        mon.dump_rep('optimizer', net.optim['optimizer'])
        if net.optim['scheduler']:
            mon.dump_rep('scheduler', net.optim['scheduler'])

        states = {
            'model_state_dict': net.state_dict(),
            'opt_state_dict': net.optim['optimizer'].state_dict()
        }
        if net.optim['scheduler']:
            states['scheduler_state_dict'] = net.optim['scheduler'].state_dict()

        mon.schedule(mon.dump, beginning=False, name='training.pt', obj=states, type='torch', keep=5)
        print('Training...')
    else:
        mon = nnt.Monitor(current_folder=checkpoint_folder, print_freq=print_freq, num_iters=len(train_data) // bs,
                          use_tensorboard=True)
        states = mon.load('training.pt', type='torch')
        mon.set_iter(mon.get_epoch() * len(train_data) // bs)

        net.load_state_dict(states['model_state_dict'])
        net.optim['optimizer'].load_state_dict(states['opt_state_dict'])
        if net.optim['scheduler']:
            net.optim['scheduler'].load_state_dict(states['scheduler_state_dict'])

        print('Resume from epoch %d...' % mon.get_epoch())

    mon.run_training(net, train_loader, n_epochs, val_loader, valid_freq=val_freq, reduce='mean')
    print('Training finished!')


if __name__ == '__main__':
    gin.parse_config_file(config_file)
    train_valid()
