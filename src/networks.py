from functools import partial
from neuralnet_pytorch import utils
import numpy as np
import torch as T

from ops import *

weights_init = T.nn.init.kaiming_normal_
bias_init = T.nn.init.zeros_
Conv = partial(nnt.Conv2d, weights_init=weights_init, bias_init=bias_init)


def normalized_chamfer_loss(pred, gt, reduce='mean', normalized=False):
    if normalized:
        max_dist, _ = T.max(T.sqrt(T.sum(gt ** 2., -1)), -1)
        origin = T.mean(gt, -2)
        pred = (pred - origin) / max_dist
        gt = (gt - origin) / max_dist

    loss = nnt.chamfer_loss(pred, gt, reduce=reduce)
    return loss if reduce == 'sum' else loss * 3000.


class _PCEncoderMethods:
    def _project_old(self, img_feats, xs, ys):
        out = []
        for i in range(list(img_feats.shape)[0]):
            x, y, img_feat = xs[i], ys[i], img_feats[i]
            x1, y1 = T.floor(x), T.floor(y)
            x2, y2 = T.ceil(x), T.ceil(y)
            q11 = img_feat[..., x1.long(), y1.long()].cuda()
            q12 = img_feat[..., x1.long(), y2.long()].cuda()
            q21 = img_feat[..., x2.long(), y1.long()].cuda()
            q22 = img_feat[..., x2.long(), y2.long()].cuda()

            weights = ((x2 - x) * (y2 - y)).unsqueeze(0)
            q11 *= weights

            weights = ((x - x1) * (y2 - y)).unsqueeze(0)
            q21 *= weights

            weights = ((x2 - x) * (y - y1)).unsqueeze(0)
            q12 *= weights

            weights = ((x - x1) * (y - y1)).unsqueeze(0)
            q22 *= weights
            out.append(q11 + q12 + q21 + q22)
        return T.stack(out).transpose(2, 1)

    def _project(self, img_feats, xs, ys):
        x, y = xs.flatten(), ys.flatten()
        idb = T.arange(img_feats.shape[0], device=img_feats.device)
        idb = idb[None].repeat(xs.shape[1], 1).t().flatten().long()

        x1, y1 = T.floor(x), T.floor(y)
        x2, y2 = T.ceil(x), T.ceil(y)
        q11 = img_feats[idb, :, x1.long(), y1.long()].to(img_feats.device)
        q12 = img_feats[idb, :, x1.long(), y2.long()].to(img_feats.device)
        q21 = img_feats[idb, :, x2.long(), y1.long()].to(img_feats.device)
        q22 = img_feats[idb, :, x2.long(), y2.long()].to(img_feats.device)

        weights = ((x2 - x) * (y2 - y)).unsqueeze(1)
        q11 *= weights

        weights = ((x - x1) * (y2 - y)).unsqueeze(1)
        q21 *= weights

        weights = ((x2 - x) * (y - y1)).unsqueeze(1)
        q12 *= weights

        weights = ((x - x1) * (y - y1)).unsqueeze(1)
        q22 *= weights
        out = q11 + q12 + q21 + q22
        return out.view(img_feats.shape[0], -1, img_feats.shape[1])

    def get_projection(self, img_feat, pc):
        _, _, h_, w_ = tuple(img_feat.shape)
        X, Y, Z = pc[..., 0], pc[..., 1], pc[..., 2]
        h = 248. * Y / Z + 111.5
        w = 248. * -X / Z + 111.5
        h = T.clamp(h, 0., 223.)
        w = T.clamp(w, 0., 223.)

        x = (h / (223. / (h_ - 1.))).requires_grad_(False)
        y = (w / (223. / (w_ - 1.))).requires_grad_(False)
        feats = self._project(img_feat, x, y)
        return feats

    def transform(self, pc_feat, img_feat, fc):
        pc_feat = (pc_feat - T.mean(pc_feat, -1, keepdim=True)) / T.sqrt(T.var(pc_feat, -1, keepdim=True) + 1e-8)
        mean, var = T.mean(img_feat, (2, 3)), T.var(T.flatten(img_feat, 2), 2)
        output = (pc_feat + nnt.utils.dimshuffle(mean, (0, 'x', 1))) * T.sqrt(
            nnt.utils.dimshuffle(var, (0, 'x', 1)) + 1e-8)
        return fc(output)

    def _forward(self, img_feats, init_pc):
        features = []
        if self.adain:
            pc_feats = []
            # centered_pc = init_pc - T.mean(init_pc, -1, keepdim=True)
            out_feat = init_pc
            for block in self.blocks.children():
                out_feat = block(out_feat)
                pc_feats.append(out_feat)

            features += [self.transform(pc_feat, img_feat, fc) for pc_feat, img_feat, fc in
                         zip(pc_feats, img_feats, self.fcs.children())]

        if self.projection:
            features += [self.get_projection(img_feat, init_pc) for img_feat in img_feats]

        features += [init_pc]
        pc_feats_trans = self.concat(features)
        return pc_feats_trans

    def _output_shape(self):
        modules = list(self.children())
        return modules[-1].output_shape


class _ImageEncoderMethods:
    def _forward(self, input):
        feats = []
        output = input
        for block in self.children():
            output = block(output)
            feats.append(output)
        return feats

    def _output_shape(self):
        modules = list(self.children())
        return modules[-1].output_shape


class PointCloudEncoder(nnt.Module, _PCEncoderMethods):
    def __init__(self, pc_shape, out_features, activation='relu', adain=True, projection=True):
        super().__init__(pc_shape)
        self.adain = adain
        self.projection = projection

        if self.adain:
            self.blocks = nnt.Sequential(input_shape=pc_shape)
            for i, out_features_ in enumerate(out_features):
                block = nnt.Sequential(input_shape=self.blocks.output_shape)
                block.add_module('fc1', nnt.FC(block.output_shape, out_features_, activation=activation))
                block.add_module('fc2', nnt.FC(block.output_shape, out_features_, activation=activation))
                self.blocks.add_module('block%d' % i, block)

            fcs = [nnt.FC(block.output_shape, out_features_) for block, out_features_ in zip(self.blocks, out_features)]
            self.fcs = nnt.Sequential(*fcs, input_shape=None)

        mul = self.adain + self.projection
        self.concat = nnt.Lambda(lambda x: T.cat(x, -1),
                                 input_shape=[pc_shape[:-1] + (dim,) for dim in out_features] * mul + [list(pc_shape)],
                                 output_shape=pc_shape[:-1] + (mul * sum(out_features) + pc_shape[2],))

    def forward(self, img_feats, init_pc):
        return super()._forward(img_feats, init_pc)

    @property
    @utils.validate
    def output_shape(self):
        return super()._output_shape()


class CNN18Encoder(nnt.Module, _ImageEncoderMethods):
    def __init__(self, input_shape, activation='relu'):
        super().__init__(input_shape)

        self.block1 = nnt.Sequential(input_shape=input_shape)
        self.block1.add_module('conv1', Conv(self.block1.output_shape, 16, 3, activation=activation))
        self.block1.add_module('conv2', Conv(self.block1.output_shape, 16, 3, activation=activation))
        self.block1.add_module('conv3', Conv(self.block1.output_shape, 32, 3, stride=2, activation=activation))
        self.block1.add_module('conv4', Conv(self.block1.output_shape, 32, 3, activation=activation))
        self.block1.add_module('conv5', Conv(self.block1.output_shape, 32, 3, activation=activation))
        self.block1.add_module('conv6', Conv(self.block1.output_shape, 64, 3, stride=2, activation=activation))
        self.block1.add_module('conv7', Conv(self.block1.output_shape, 64, 3, activation=activation))
        self.block1.add_module('conv8', Conv(self.block1.output_shape, 64, 3, activation=activation))

        self.block3 = nnt.Sequential(input_shape=self.block1.output_shape)
        self.block3.add_module('conv2', Conv(self.block3.output_shape, 128, 3, stride=2, activation=activation))
        self.block3.add_module('conv3', Conv(self.block3.output_shape, 128, 3, activation=activation))
        self.block3.add_module('conv4', Conv(self.block3.output_shape, 128, 3, activation=activation))

        self.block4 = nnt.Sequential(input_shape=self.block3.output_shape)
        self.block4.add_module('conv3', Conv(self.block4.output_shape, 256, 5, stride=2, activation=activation))
        self.block4.add_module('conv4', Conv(self.block4.output_shape, 256, 3, activation=activation))
        self.block4.add_module('conv5', Conv(self.block4.output_shape, 256, 3, activation=activation))

        self.block5 = nnt.Sequential(input_shape=self.block4.output_shape)
        self.block5.add_module('conv3', Conv(self.block5.output_shape, 512, 5, stride=2, activation=activation))
        self.block5.add_module('conv4', Conv(self.block5.output_shape, 512, 3, activation=activation))
        self.block5.add_module('conv5', Conv(self.block5.output_shape, 512, 3, activation=activation))
        self.block5.add_module('conv6', Conv(self.block5.output_shape, 512, 3, activation=activation))
        self.blocks = [self.block1, self.block3, self.block4, self.block5]

    def forward(self, input):
        return super()._forward(input)

    @property
    @utils.validate
    def output_shape(self):
        return super()._output_shape()


class PointCloudDecoder(nnt.Sequential):
    def __init__(self, input_shape, activation='relu'):
        super().__init__(input_shape=input_shape)

        self.add_module('conv2', nnt.FC(self.output_shape, 512, activation=activation))
        self.add_module('conv3', nnt.FC(self.output_shape, 256, activation=activation))
        self.add_module('conv4', nnt.FC(self.output_shape, 128, activation=activation))
        self.add_module('conv6', nnt.FC(self.output_shape, 3, activation=None))


class PointCloudResDecoder(nnt.Sequential):
    def __init__(self, input_shape, activation='relu'):
        super().__init__(input_shape=input_shape)

        self.add_module('conv2', ResFC(self.output_shape, 512, activation=activation))
        self.add_module('conv3', ResFC(self.output_shape, 256, activation=activation))
        self.add_module('conv4', ResFC(self.output_shape, 128, activation=activation))
        self.add_module('conv6', nnt.FC(self.output_shape, 3, activation=None))


class PointCloudGraphXDecoder(nnt.Sequential):
    def __init__(self, input_shape, activation='relu'):
        super().__init__(input_shape=input_shape)

        self.add_module('conv2', GraphXConv(self.output_shape, 512, activation=activation))
        self.add_module('conv3', GraphXConv(self.output_shape, 256, activation=activation))
        self.add_module('conv4', GraphXConv(self.output_shape, 128, activation=activation))
        self.add_module('conv6', nnt.FC(self.output_shape, 3, activation=None))


class PointCloudResGraphXDecoder(nnt.Sequential):
    def __init__(self, input_shape, activation='relu'):
        super().__init__(input_shape=input_shape)

        self.add_module('conv2', ResGraphXConv(self.output_shape, 512, activation=activation))
        self.add_module('conv3', ResGraphXConv(self.output_shape, 256, activation=activation))
        self.add_module('conv4', ResGraphXConv(self.output_shape, 128, activation=activation))
        self.add_module('conv6', nnt.FC(self.output_shape, 3, activation=None))


class PointCloudResGraphXUpDecoder(nnt.Sequential):
    def __init__(self, input_shape, activation='relu'):
        super().__init__(input_shape=input_shape)

        self.add_module('conv2', ResGraphXConv(self.output_shape, 512, num_instances=self.output_shape[1] * 2,
                                               activation=activation))
        self.add_module('conv3', ResGraphXConv(self.output_shape, 256, num_instances=self.output_shape[1] * 2,
                                               activation=activation))
        self.add_module('conv4', ResGraphXConv(self.output_shape, 128, num_instances=self.output_shape[1] * 2,
                                               activation=activation))
        self.add_module('conv6', nnt.FC(self.output_shape, 3, activation=None))


class PointCloudResLowRankGraphXUpDecoder(nnt.Sequential):
    def __init__(self, input_shape, activation='relu', decimation=.5):
        super().__init__(input_shape=input_shape)

        self.add_module('conv2', ResLowRankGraphXConv(self.output_shape, 512, num_instances=self.output_shape[1] * 2,
                                                      rank=int(decimation * self.output_shape[1]), activation=activation))
        self.add_module('conv3', ResLowRankGraphXConv(self.output_shape, 256, num_instances=self.output_shape[1] * 2,
                                                      rank=int(decimation * self.output_shape[1]), activation=activation))
        self.add_module('conv4', ResLowRankGraphXConv(self.output_shape, 128, num_instances=self.output_shape[1] * 2,
                                                      rank=int(decimation * self.output_shape[1]), activation=activation))
        self.add_module('conv6', nnt.FC(self.output_shape, 3, activation=None))


class PointcloudDeformNet(nnt.Net, nnt.Module):
    def __init__(self, input_shape, pc_shape, img_enc, pc_enc, pc_dec, activation='relu', adain=True, projection=True,
                 optimizer=None, scheduler=None, **kwargs):
        super().__init__(input_shape=input_shape)

        self.img_enc = img_enc(input_shape, activation)
        self.pc_enc = pc_enc(pc_shape, [block.output_shape[1] for block in self.img_enc.blocks], activation,
                             adain=adain, projection=projection)
        self.pc = pc_dec(self.pc_enc.output_shape, activation=activation)

        self.optim['optimizer'] = T.optim.Adam(self.trainable, 1e-4, weight_decay=0) if optimizer is None \
            else optimizer(self.trainable)
        self.optim['scheduler'] = scheduler(self.optim['optimizer']) if scheduler else None
        self.kwargs = kwargs

    def forward(self, input, init_pc):
        img_feats = self.img_enc(input)
        pc_feats = self.pc_enc(img_feats, init_pc)
        return self.pc(pc_feats)

    @property
    @utils.validate
    def output_shape(self):
        modules = list(self.children())
        return modules[-1].output_shape

    def regularization(self):
        return sum(T.sum(w ** 2) for w in self.regularizable)

    def train_procedure(self, init_pc, input, gt_pc, reduce='mean', normalized=False):
        pred_pc = self(input, init_pc)

        loss = sum([normalized_chamfer_loss(pred[None], gt[None], reduce=reduce, normalized=normalized) for pred, gt in zip(pred_pc, gt_pc)]) / len(
            gt_pc) if isinstance(gt_pc, (list, tuple)) else normalized_chamfer_loss(pred_pc, gt_pc, reduce=reduce, normalized=normalized)
        return loss

    def eval_procedure(self, init_pc, input, gt_pc, reduce='mean', normalized=False):
        pred_pc = self(input, init_pc)

        loss = sum([normalized_chamfer_loss(pred[None], gt[None], reduce=reduce, normalized=normalized) for pred, gt in zip(pred_pc, gt_pc)]) / len(
            gt_pc) if isinstance(gt_pc, (list, tuple)) else normalized_chamfer_loss(pred_pc, gt_pc, reduce=reduce, normalized=normalized)
        self.stats['eval']['scalars']['eval_chamfer'] = nnt.utils.to_numpy(loss)
        del loss

    def learn(self, init_pc, input, gt_pc, *args, **kwargs):
        self.train(True)
        self.optim['optimizer'].zero_grad()
        loss = self.train_procedure(init_pc, input, gt_pc, reduce='mean')
        if not (T.isnan(loss) or T.isinf(loss)):
            loss.backward()
            self.optim['optimizer'].step()

        self.stats['train']['scalars']['chamfer'] = nnt.utils.to_numpy(loss)
        del loss
