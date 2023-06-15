import torch

def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    return loss

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False, pc_distance=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1, 1, M, 1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1, N, 1, 1)
    pc_diff = pc1_expand_tile - pc2_expand_tile

    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1)  # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1)  # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff ** 2, dim=-1)  # (B,N,M)
    dist1, idx1 = torch.min(pc_dist, dim=2)  # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1)  # (B,M)
    if pc_distance:
        return pc_dist
    else:
        return dist1, idx1, dist2, idx2

def index_point(points, index):
    device = points.device
    B = points.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    index_points = points[batch_indices, index, :]
    return index_points