import numpy as np
import argparse

from networks import Pixel2Pointcloud
from data_loader import ShapeNet, collate
from train import *

parser = argparse.ArgumentParser('GraphX-convolution')
parser.add_argument('path', type=str, help='path to the ShapeNet train/test spit')
parser.add_argument('--no_graphx', action='store_true', help='not use graphx')
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint to a pretrained model folder')
parser.add_argument('-v', '--version', type=int, default=-1, help='version of the checkpoint')
args = parser.parse_args()


def test_each_category():
    mon = Monitor(current_folder=checkpoint_folder, print_freq=1)
    states = mon.load('training.pt', type='torch', version=version)
    net = Pixel2Pointcloud(3 if color_img else 1, n_points, use_graphx=not args.no_graphx)
    print(net)
    net.load_state_dict(states['model_state_dict'])
    net.eval()

    for file_cat in file_list:
        folder = file_cat
        if version != -1:
            folder += '_%d' % version

        if not os.path.exists(os.path.join(mon.current_folder, folder)):
            os.mkdir(os.path.join(mon.current_folder, folder))

        test_data = ShapeNet([file_cat], path=path, grayscale=not color_img, type='test', n_points=n_points, metadata=True)
        test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=10, collate_fn=collate)

        mon.set_iter(0)
        mon.clear_num_stats('/test chamfer')
        print('Testing %s...' % file_cat)
        with T.set_grad_enabled(False):
            for itt, batch in enumerate(test_loader):
                init_pc, image, gt_pc, metadata = batch
                if T.cuda.is_available():
                    init_pc = init_pc.cuda()
                    image = image.cuda()
                    gt_pc = [pc.cuda() for pc in gt_pc] if isinstance(gt_pc, (list, tuple)) else gt_pc.cuda()

                loss_dict, pred_pc = net.loss(image, init_pc, gt_pc, reduce='mean')
                pred_pc = to_numpy(pred_pc)
                gt_pc = bulk_to_numpy(gt_pc)
                loss = to_numpy(loss_dict['chamfer']) / 3.
                with mon:
                    mon.plot(folder + '/test chamfer', loss)
                    for ii in range(pred_pc.shape[0]):
                        mon.dump(folder + '/%s gt.xyz' % metadata[ii], gt_pc[ii], 'txt', delimiter=' ')
                        mon.dump(folder + '/%s pred %.4f.xyz' % (metadata[ii], loss), pred_pc[ii], 'txt', delimiter=' ')
    print('Testing finished!')


if __name__ == '__main__':
    test_each_category()
