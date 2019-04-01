import torch as T


def to_numpy(x):
    return x.cpu().detach().data.numpy()


def to_cuda(x):
    return T.from_numpy(x).cuda()


def bulk_to_numpy(xs):
    return tuple([to_numpy(x) for x in xs])


def batch_pairwise_dist(x, y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = T.bmm(x, x.transpose(2, 1))
    yy = T.bmm(y, y.transpose(2, 1))
    zz = T.bmm(x, y.transpose(2, 1))

    if T.cuda.is_available():
        dtype = T.cuda.LongTensor
    else:
        dtype = T.LongTensor

    diag_ind_x = T.arange(0, num_points_x).type(dtype)
    diag_ind_y = T.arange(0, num_points_y).type(dtype)
    # brk()
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P


def chamfer_loss(xyz1, xyz2, reduce='sum'):
    """
    The Pytorch code is adapted from https://github.com/345ishaan/DenseLidarNet/blob/master/code/chamfer_loss.py

    :param xyz1: a point cloud of shape (b, n1, 3)
    :param xyz2: a point cloud of shape (b, n2, 3)
    :param reduce: 'mean' or 'sum'. Default is 'sum'
    :return: the Chamfer distance between the two point clouds
    """
    assert reduce in ('mean', 'sum'), 'Unknown reduce method'
    reduce = T.sum if reduce == 'sum' else T.mean
    P = batch_pairwise_dist(xyz1, xyz2)
    dist2, _ = T.min(P, 1)
    dist1, _ = T.min(P, 2)
    loss_2 = reduce(dist2)
    loss_1 = reduce(dist1)
    return loss_1 + loss_2
