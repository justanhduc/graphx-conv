import torch as T
from torch.utils.data import Dataset
import numpy as np
import os
import pickle as pkl


def sample_spherical(n_points):
    vec = np.random.rand(n_points, 3) * 2. - 1.
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    pc = vec * .3 + np.array([[6.462339e-04,  9.615256e-04, -7.909229e-01]])
    return pc.astype('float32')


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def collate(batch):
    data = [b for b in zip(*batch)]
    if len(data) == 3:
        init_pc, imgs, gt_pc = data
    elif len(data) == 4:
        init_pc, imgs, gt_pc, metadata = data
    else:
        raise ValueError('Unknown data values')

    init_pc = T.from_numpy(np.array(init_pc)).requires_grad_(False)
    imgs = T.from_numpy(np.array(imgs)).requires_grad_(False)
    gt_pc = [T.from_numpy(pc).requires_grad_(False) for pc in gt_pc]
    return (init_pc, imgs, gt_pc) if len(data) == 3 else (init_pc, imgs, gt_pc, metadata)


class ShapeNet(Dataset):
    def __init__(self, file_list, path, grayscale=None, type='train', n_points=2000, metadata=False, **kwargs):
        assert type in ('train', 'valid', 'test')
        self.n_points = n_points
        self.grayscale = grayscale
        self.file_list = file_list
        self.path = path
        self.type = type if type in ('train', 'test') else 'test'
        self.metadata = metadata
        self.num_vals = kwargs.pop('num_vals', 30)
        self.pkl_list = []
        self.sample_weights = []
        for folder in file_list:
            file_path = os.listdir(os.path.join(path, folder, self.type))
            if type == 'valid':
                idx = np.random.randint(len(file_path), size=self.num_vals // len(file_list))
                file_path = [file_path[i] for i in idx]

            file_path = [os.path.join(self.path, folder, self.type, f) for f in file_path]
            self.pkl_list.extend(file_path)
            self.sample_weights.extend([1 / len(file_path)] * len(file_path))

    def __len__(self):
        return len(self.pkl_list)

    def __getitem__(self, idx):
        pkl_path = self.pkl_list[idx]
        contents = pkl.load(open(pkl_path, 'rb'), encoding='latin1')
        img = rgb2gray(contents[0])[..., None] if self.grayscale else contents[0]
        img = (np.transpose(img / 255.0, (2, 0, 1)) - .5) * 2
        pc = np.array(contents[1], 'float32')[:, :3]
        pc -= np.mean(pc, 0, keepdims=True)
        item = (sample_spherical(self.n_points), np.array(img, 'float32'), pc)
        if self.metadata:
            metadata = pkl_path.split('\\')[-1][:-4]
            item += (metadata,)

        return item
