import torch
import torch.nn as nn
from torch.autograd import Variable


def warp(x, flow, differentiable=True):
    """
    Warp x according to the optical flow.

    Parameters
    ----------
    x : Tensor
        Tensor to be warped.
    flow : Tensor
        Optical flow.
    differentiable : bool
        Determin, whether to calculate gradient in the warping operation

    Returns
    -------
    x_out : Tensor (on CUDA)
        Warped tensor.
    """
    B, C, H, W = x.size()
    flow = flow.cuda()
    x = x.cuda()

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    if differentiable:
        vgrid = Variable(grid) + flow
    else:
        vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.ones(x.size()).cuda()

    if differentiable:
        mask = Variable(mask)

    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    x_out = output * mask
    return x_out
