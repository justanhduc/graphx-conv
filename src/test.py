import argparse

parser = argparse.ArgumentParser('GraphX-convolution')
parser.add_argument('config_file', type=str, help='config file to dictate training/testing')
parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu number')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from torch.utils.data import DataLoader
import neuralnet_pytorch.gin_nnt as gin

from networks import *
from data_loader import ShapeNet, collate

config_file = args.config_file
bs = 150

gin.external_configurable(CNN18Encoder, 'cnn18_enc')
gin.external_configurable(PointCloudEncoder, 'pc_enc')
gin.external_configurable(PointCloudDecoder, 'pc_dec')
gin.external_configurable(PointCloudResDecoder, 'pc_resdec')
gin.external_configurable(PointCloudResGraphXUpDecoder, 'pc_upresgraphxdec')
gin.external_configurable(PointCloudResLowRankGraphXUpDecoder, 'pc_upreslowrankgraphxdec')


@gin.configurable('GraphX')
def test_each_category(data_root, checkpoint_folder, img_enc, pc_enc, pc_dec, color_img=False, n_points=250, **kwargs):
    mon = nnt.Monitor(current_folder=checkpoint_folder, print_freq=1)
    states = mon.load('training.pt', type='torch')
    net = PointcloudDeformNet((bs,) + (3 if color_img else 1, 224, 224), (bs, n_points, 3), img_enc, pc_enc, pc_dec)
    print(net)
    net.load_state_dict(states['model_state_dict'])
    net.eval()

    for file_cat in os.listdir(data_root):
        if not os.path.exists(os.path.join(mon.current_folder, file_cat)):
            os.mkdir(os.path.join(mon.current_folder, file_cat))

        test_data = ShapeNet([file_cat], path=data_root, grayscale=not color_img, type='test', n_points=n_points)
        test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=10, collate_fn=collate)

        mon.set_iter(0)
        mon.clear_scalar_stats(file_cat + '/test chamfer')
        print('Testing...')
        with T.set_grad_enabled(False):
            for itt, batch in enumerate(test_loader):
                init_pc, image, gt_pc = batch
                if nnt.cuda_available:
                    init_pc = init_pc.cuda()
                    image = image.cuda()
                    gt_pc = [pc.cuda() for pc in gt_pc] if isinstance(gt_pc, (list, tuple)) else gt_pc.cuda()

                pred_pc = net(image, init_pc)
                loss = sum([normalized_chamfer_loss(pred[None], gt[None]) for pred, gt in zip(pred_pc, gt_pc)]) / (
                            3. * len(gt_pc))
                loss = nnt.utils.to_numpy(loss)
                with mon:
                    mon.plot(file_cat + '/test chamfer', loss)
    print('Testing finished!')


if __name__ == '__main__':
    gin.parse_config_file(config_file)
    test_each_category()
