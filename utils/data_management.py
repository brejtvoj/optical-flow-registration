import os
import numpy as np
from PIL import Image
import csv
import argparse

def load_csv(path):
    """
    Load landmarks of the given image.

    Parameters
    ----------
    path : string
        Path to the given csv file, without extenstion.

    Returns
    -------
    landmarks : list
        List of landmarks in the shape of (y_coord, x_coord).
    """
    landmarks = list()

    with open(path + '.csv', newline='') as csvfile:
        sheet = csv.reader(csvfile)
        for i, line in enumerate(sheet):
            # Ignore header
            if i == 0:
                continue

            x_cord = float(line[1])
            y_cord = float(line[2])
            landmarks.append((y_cord, x_cord))

    return landmarks


def load_image(path, extension='.png'):
    """
    Return a loaded image.

    Parameters
    ----------
    path : string
        Path to the given image,  without extension.
    extension : string
        File type of the image.

    Returns
    -------
    image : ndarray
        Loaded image.
    """
    image = np.array(Image.open(path + extension))
    return image


def load_image_and_landmarks(path):
    """
    Return a loaded image and landmarks.

    Parameters
    ----------
    path : string
        Path to the given image and landmarks without extension.

    Returns
    -------
    image : ndarray
        Loaded image.
    lm_np: ndarray
        Loaded landmarks inside a numpy array.
    """
    image = load_image(path)
    landmarks = load_csv(path)

    # Convert landmarks to ndarray
    lm_np = np.zeros((len(landmarks), 2))
    for i in range(len(landmarks)):
        lm_np[i, :] = np.array(landmarks[i])

    return image, lm_np


def get_all_paths(path):
    """
    Return all posible image pairs in the given folder.

    Folder must have the same strucutre as the ANHIR dataset.

    Parameters
    ----------
    path : string
        Path to the folder, containg the datase.

    Returns
    -------
    pairs : list of tuples
        List containing paths to all possible image pairs.
    """
    directories = os.listdir(path)
    pairs = list()

    for directory in directories:
        items = os.listdir(path + directory)
        items = items[0:len(items):2]
        for i in range(len(items)):
            items[i] = items[i][:-4]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                path_1 = path + directory + '/' + items[i]
                path_2 = path + directory + '/' + items[j]
                pairs.append((path_1, path_2))
    return pairs


def get_paths(path):
    """
    Randomly select an image pair.

    Parameters
    ----------
    path : string
        Folder with the dataset.

    Returns
    -------
    path1 : string
        Path to the first image.
    path2 : string
    Path to the second image.
    """
    directories = os.listdir(path)
    path = path + np.random.choice(directories) + '/'

    items = os.listdir(path)

    path1 = path + np.random.choice(items, replace=False)[:-4]
    path2 = path + np.random.choice(items, replace=False)[:-4]

    while path1 == path2:
        path2 = path + np.random.choice(items, replace=False)[:-4]

    return path1, path2


def save_flow(uv, filename, v=None):
    """ Write optical flow to file.
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    TAG_CHAR = np.array([202021.25], np.float32)
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def save_image(image, filename, const=255):
    """
    Save image.

    Parameters
    ----------
    image : ndarray
        Image to be saved.
    filename : string
        Path to the save location
    const : float
        Scale factor.
    """
    img = Image.fromarray((const * image).astype('uint8')).convert('RGB')
    img.save(filename)


def save_mask(self, mask, filename):
    """
    Save mask.

    Parameters
    ----------
    mask : ndarray
        Mask to be saved.
    filename : string
        Path to the save location
    """
    np.save(filename, mask)


def warp_landmarks_from_sample(flow, landmarks):
    """
    Warp landmarks by finding the corresponding point in the transformed grid.

    Parameters
    ----------
    flow : Tensor
        Optical flow.
    landmarks : list of tuples
        List of the original landmarks.

    Returns
    -------
    landmarks_out : list of tuples
        List of warped landmarks.

    """
    flow = flow.detach().squeeze().cpu().numpy()
    # Generate grid
    y = np.arange(flow.shape[1])
    x = np.arange(flow.shape[2])
    X, Y = np.meshgrid(x, y)
    grid = np.zeros_like(flow)
    grid[0, :, :] = X
    grid[1, :, :] = Y

    # Transform grid
    grid_t = grid + flow

    landmarks_out = list()

    # Find the closest point in the new grid for all landmarks
    for landmark in landmarks:
        point = np.array(landmark)
        dx = grid_t[0, :, :] - point[1]
        dy = grid_t[1, :, :] - point[0]
        dist = np.abs(dx) + np.abs(dy)
        lm_i = np.argmin(dist)
        lm_i = np.unravel_index(lm_i, flow.shape[1:])
        landmarks_out.append(lm_i)
    return landmarks_out


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', default='anhir', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, nargs='+', default=[512, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=7)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--in_channels', type=int, default=3)
    args_out = parser.parse_args()
    return args_out

def gen_args():
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
    parser.add_argument('--iters', type=int, default=7)
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
