import numpy as np
import scipy.signal as sig

import torch
import torch.nn.functional as F


# Set the computation device globally to either the CPU of the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_kernel(size=9, sigma=10, channels=3):
    """
    Return a Ricker wavelet in the correct shape for the PyTorch convolution.

    Parameters
    ----------
    size : int
        Size of the kernel.
    sigma : float
        Width parameter of the Ricker wavelet.
    channels : int
        Number of channels of the input image.

    Returns
    -------
    kernel : Tensor
        PyTorch representation of the Ricker Wavelet.
    """
    # Generate 2D kernel
    kernel_channel = sig.ricker(size, sigma)
    kernel_channel = np.add.outer(kernel_channel, kernel_channel)

    # Generate 4D convolutional kernel
    kernel = np.zeros((1, channels, size, size))
    for c in range(channels):
        kernel[0, c, :, :] = kernel_channel

    return torch.from_numpy(kernel).float()


def generate_gaussian(size=11, sigma=5):
    """
    Return a Gaussian in the correct shape for the PyTorch convolution.

    Parameters
    ----------
    size : int
        Size of the kernel.
    sigma : float
        Variance of the Gaussian.
    channels : int
        Number of channels of the input image.

    Returns
    -------
    kernel : Tensor
        PyTorch representation of the Gaussian filter.
    """
    x = np.linspace(-(size // 2), (size // 2), size)
    x /= np.sqrt(2)*sigma
    x2 = x**2

    kernel_channel = np.exp(- x2[:, None] - x2[None, :])
    kernel_channel = kernel_channel / kernel_channel.sum()

    kernel = np.zeros((1, 1, size, size))
    kernel[0, 0, :, :] = kernel_channel

    return torch.from_numpy(kernel).float()


def filter_image(image, kernel, pad=1):
    """
    Prepare and filter the image using PyTorch convolution.

    Parameters
    ----------
    image : ndarray
        Input image.
    kernel : Tensor
        Kernel of the convolution.
    pad : int
        Padding value at the edges.

    Returns
    -------
    image_out : Tensor
       Image filtered with the given kernel.
    """
    image = torch.from_numpy(image).to(device).float()
    kernel = kernel.to(device).float()

    image_out = F.conv2d(image, kernel, padding=pad)

    return image_out


def dilation_filter(size):
    """
    Return a dilation in the correct shape for the PyTorch convolution.

    Parameters
    ----------
    size : int
        Size of the kernel.

    Returns
    -------
    kernel : Tensor
        PyTorch representation of the dilation filter.
    """
    dil_filter = np.ones((size, size)) / size**2

    kernel = np.zeros((1, 1, size, size))
    kernel[0, 0, :, :] = dil_filter

    return torch.from_numpy(kernel).float()


def dilation(image, size, repeats=1):
    """
    Dilate the image with the kernel of given size n-times.

    Parameters
    ----------
    image : ndarray
        Input image.
    size : int
        Size of the dilation filter.
    repeats : int
        Number of repeats of the dilation.

    Returns
    -------
    image : Tensor
        Dilated image.
    """
    dil_filter = dilation_filter(size)

    # Torch setup
    zeros = torch.zeros_like(image).to(device)
    ones = torch.ones_like(image).to(device)

    for _ in range(repeats):
        image = filter_image(image, dil_filter, pad=5)
        image = torch.where(image <= 0.5, ones, zeros)
    return image


def blur(image, strenght=1, repeats=1, pad=4):
    """
    Blur an image with the Gaussian filter n-times.

    Parameters
    ----------
    image : ndarray
        Input image.
    strenght : int
        Variable controling the shape and values of the Gaussian filter.
    repeats : int
        Number of repeats of the blurring.
    pad : int
        Padding value at the edges.

    Returns
    -------
    image : Tensor
        Blurred image.
    """
    kernel = generate_gaussian(3 * strenght, strenght)

    for _ in range(repeats):
        image = filter_image(image, kernel, pad)

    return image


def segment(image, hp_kernel, lp_kernel, threshold=None):
    """
    Segment the image by repeated dilation and blurring.

    Parameters
    ----------
    image : ndarray
        Input image.
    hp_kernel : Tensor
        High-pass kernel.
    lp_kernel : Tensor
        Low-pass kernel.
    threshold : float
        Value which divides background and foreground.

    Returns
    -------
    mask : ndarray
        Foreground mask [0, 1].
    """
    # Prepare image
    image = torch.from_numpy(image).permuter(2, 0, 1).unsqueeze(dim=0)

    # High-pass filtering
    image_out = filter_image(image, hp_kernel, pad=4)

    # Normalize into the [0, 1] range
    img_min = torch.min(image_out)
    img_max = torch.max(image_out)
    image_out = (image_out - img_min) / (img_max - img_min)

    if not threshold:
        threshold = torch.mean(image_out) * 1.03

    # Torch preparation
    zeros = torch.zeros_like(image_out).to(device)
    ones = torch.ones_like(image_out).to(device)

    # Threshold filtered image and convert to {0, 1}
    image_out = torch.where(image_out >= threshold, ones, zeros)
    image_out = image_out.cpu().numpy()

    # Low-pass filtering
    image_out = filter_image(image_out, lp_kernel, pad=5)

    # Thresholding
    image_out = torch.where(image_out <= 0.5, ones, zeros)

    # Repeated dilation
    image_out = dilation(image_out, 11, repeats=10)

    # Blur edges of the mask
    image_out = blur(image_out, strenght=3, repeats=2)

    mask = image_out.cpu().numpy().squeeze()
    return mask


def segment_wrapper(image):
    """
    Segmented the image with predefined parameters.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    mask : ndarray
        Foreground mask [0, 1].
    """
    # Generate kernels
    hp_kernel = generate_kernel(channels=3)
    lp_kernel = generate_gaussian()

    # Segment the image
    image_seg = segment(image, hp_kernel, lp_kernel)

    return image_seg
