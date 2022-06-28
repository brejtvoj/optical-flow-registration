import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from numpy import ndarray


import argparse


from data_managment import load_image, load_csv, warp_landmarks_from_sample

# Ideal index of flow updates - hyperparameter
ITER = 6


def register(path1, path2, model_type, model_path, coeff,
             image1=None, image2=None, landmarks1=None, landmarks2=None,
             div=8):
    """
    Register images on the global scale.

    Downsample and pad images to have the same size. Low-resolution flow
    is interpolated and scaled to the original resolution.

    Parameters
    ----------
    path1 : string
        Location of the original image.
    path2 : string
        Location of the target image.
    model_type : nn.Module
        Determin, which architecture to use (RAFT / GMA).
    model_path : string
        Location of the model weights.
    coeff : float
        Scale factor.
    image1 : ndarray
        Possible skip of loading the original image.
    image2 : ndarray
        Possible skip of loading the target image.
    landmarks1 : list of tuples
        Landmarks of the first image.
    landmarks2 : list of tuples
        Landmarks of the second image.
    div : int
        Value, which should the resulting shape be divisible by.

    Returns
    -------
    flow : Tensor
        Global optical flow.
    image1 : Tensor
        Original image.
    image2 : Tensor
        Target image.
    lmW : list of tuples
        Landmarks of the warped image.
    lm2 : list of tuples
        Landmarks of the target image.
    lm1_unwarped : list of tuples
        Landmarks of the original image.

    """
    shape_original = load_image(path1).shape[:2]
    blob = prepare_image_and_landmarks(path1, path2,
                                       coeff, div, image1, image2,
                                       landmarks1, landmarks2)

    image1, image2, lm1_o, lm2_o, shape_f1, shape_f2 = blob
    lm1 = lm1_o
    lm2 = lm2_o
    model = model_type(gen_args())
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval().cuda()
    flow, flow_down = get_flow(model, image1, image2, shape_original)
    lm1_unwarped = lm1
    lmW = warp_landmarks_from_sample(flow_down[ITER], lm1)
    return flow, image1, image2, lmW, lm2, lm1_unwarped


def _register(model, image1, image2):
    """Prepare and register images."""
    if not isinstance(image1, Tensor):
        image1 = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(dim=0)
    if not isinstance(image2, Tensor):
        image1 = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(dim=0)

    flow = model(image1.float().cuda(), image2.float().cuda())
    return flow


def get_flow(model, image1, image2, shape_original):
    """Generate both the low and high resolution flow."""
    flow_downsampled = _register(model, image1, image2)
    flow = scale_flow(flow_downsampled, shape_original)
    return flow, flow_downsampled


def prepare_image_and_landmarks(path1, path2, coeff, div,
                                image1=None, image2=None,
                                landmarks1=None, landmarks2=None):
    """Interpolate and pad the images and landmarks."""
    if not isinstance(image1, ndarray):
        image1 = load_image(path1)
    if not isinstance(image2, ndarray):
        image2 = load_image(path2)
    if landmarks1 is None:
        landmarks1 = load_csv(path1)
    if landmarks2 is None:
        landmarks2 = load_csv(path2)

    shape_old1 = image1.shape[:2]
    shape_old2 = image2.shape[:2]
    y = max(image1.shape[0], image2.shape[0])
    x = max(image1.shape[1], image2.shape[1])
    y = int(div * np.ceil(coeff * y / div))
    x = int(div * np.ceil(coeff * x / div))
    shape_new = (y, x)

    imageI1 = interpolate_image(image1, shape_new)
    imageI2 = interpolate_image(image2, shape_new)

    max_len = min(len(landmarks1), len(landmarks2))
    landmarks1 = landmarks1[:max_len]
    landmarks2 = landmarks2[:max_len]
    landmarksI1 = interpolate_landmarks(landmarks1, shape_old1, shape_new)
    landmarksI2 = interpolate_landmarks(landmarks2, shape_old2, shape_new)
    return imageI1, imageI2, landmarksI1, landmarksI2, shape_old1, shape_old2


def interpolate_image(image, image_size):
    """Prepare and interpolate image."""
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(dim=0)
    image = F.interpolate(image, image_size)
    return image


def interpolate_landmarks(landmarks, shape_old, shape_new):
    """Scale landmarks to correspond to the new size."""
    center_old = np.array(shape_old) / 2
    center_new = np.array(shape_new) / 2
    y_coeff = shape_new[0] / shape_old[0]
    x_coeff = shape_new[1] / shape_old[1]
    for i in range(len(landmarks)):
        y_old = landmarks[i][0]
        x_old = landmarks[i][1]
        y_new = y_coeff * (y_old - center_old[0]) + center_new[0]
        x_new = x_coeff * (x_old - center_old[1]) + center_new[1]
        landmarks[i] = (y_new, x_new)
    return landmarks


def scale_flow(flow, shape_new):
    """Scale flow magnitudes to match the new resolution."""
    shape_old = flow[0].shape[2:]
    flow = flow[ITER]
    flow = F.interpolate(flow, (shape_new))
    flow[:, 1, :, :] *= (shape_new[0] / shape_old[0])
    flow[:, 0, :, :] *= (shape_new[1] / shape_old[0])
    return flow


def gen_args():
    """Arguments parsing"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help="name your experiment")
    parser.add_argument('--stage', default='anhir', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--output', type=str, default='checkpoints', help='output directory to save checkpoints and plots')

    parser.add_argument('--lr', type=float, default=0.0000084)
    parser.add_argument('--num_steps', type=int, default=50000-14000-15000)
    parser.add_argument('--batch_size', type=int, default=2)
    size = 320+32+16
    parser.add_argument('--image_size', type=int, nargs='+', default=[size, size])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--upsample-learn', action='store_true', default=False,
                        help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--iters', type=int, default=2)
    parser.add_argument('--val_freq', type=int, default=10000,
                        help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='printing frequency')

    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')
    parser.add_argument('--model_name', default='', help='specify model name')

    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    args = parser.parse_args()
    return args
