import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
from utils_warp import warp

import csv
import numpy as np
from scipy.stats import multivariate_normal as multivar
from PIL import Image
from skimage.exposure import match_histograms

from network import RAFTGMA
import argparse

class SplitImage():
    """Class for splitting the image into separe patches"""

    def __init__(self, patch_size=256):
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.image_size = None
        self.image1 = None
        self.image2 = None
        self.coords = None

    def load_image(self, image_path, mode, scale):
        """Load and interpolate images to have the same size."""
        imagePIL = Image.open(image_path)
        if mode == 'original':
            self.image1 = torch.from_numpy(np.array(imagePIL)).permute(2, 0, 1).numpy()
            self.image_size = self.image1.shape[1:]
            if scale is not None:
                self.image1 = torch.from_numpy(self.image1).unsqueeze(dim=0)
                self.image_size = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))
                self.image1 = F.interpolate(self.image1, self.image_size).squeeze().numpy()
        if mode == 'target':
            self.image2 = torch.from_numpy(np.array(imagePIL)).permute(2, 0, 1)
            if self.image2.shape[1:] != self.image_size:
                self.image2 = self.image2.unsqueeze(dim=0)
                self.image2 = F.interpolate(self.image2, self.image_size)
                self.image2 = self.image2.squeeze().numpy()
            else:
                self.image2 = self.image2.numpy()

    def get_split_points(self, freq=7):
        """Calculate the locations of the split points based on the frequency and the patch size."""
        y_count = np.floor(freq * self.image_size[0] / self.patch_size[0]).astype(int) + 2
        x_count = np.floor(freq * self.image_size[1] / self.patch_size[1]).astype(int) + 2
        y_splits = [i * self.patch_size[0] // freq for i in range(y_count)]
        x_splits = [i * self.patch_size[1] // freq for i in range(x_count)]
        y_splits = np.clip(y_splits, 0, self.image_size[0])
        x_splits = np.clip(x_splits, 0, self.image_size[1])
        coords = list()
        for y in range(y_count - freq):
            for x in range(x_count - freq):
                coord1 = (y_splits[y], x_splits[x])
                coord2 = (y_splits[y + freq], x_splits[x + freq])
                if coord2[0] - coord1[0] != self.patch_size[0]:
                    y_start = coord1[0] - self.patch_size[0] + coord2[0] - coord1[0]
                else:
                    y_start = coord1[0]
                if coord2[1] - coord1[1] != self.patch_size[1]:
                    x_start = coord1[1] - self.patch_size[1] + coord2[1] - coord1[1]
                else:
                    x_start = coord1[1]
                coords.append(((y_start, x_start), coord2))
        self.coords = coords

    def crop(self):
        """Crop the image into patches."""
        images_cropped1 = list()
        images_cropped2 = list()
        for coord in self.coords:
            coord1 = coord[0]
            coord2 = coord[1]
            image_crop1 = self.image1[:, coord1[0]: coord2[0], coord1[1]: coord2[1]]
            image_crop2 = self.image2[:, coord1[0]: coord2[0], coord1[1]: coord2[1]]
            images_cropped1.append(image_crop1)
            images_cropped2.append(image_crop2)
        return images_cropped1, images_cropped2

    def __call__(self, image_path1, image_path2, scale, image1=None, image2=None,
                 gray=False, matched=True):
        """Load images, crop them into small patches, and optionally modify."""
        if image1 is None:
            self.load_image(image_path1, mode='original', scale=scale)
        else:
            self.image1 = image1
        if image2 is None:
            self.load_image(image_path2, mode='target', scale=scale)
        else:
            self.image2 = image2
        if gray:
            self.image1 = torch.mean(torch.tensor(self.image1).float(), axis=0).numpy()
            self.image2 = torch.mean(torch.tensor(self.image2).float(), axis=0).numpy()

        if matched:
            if np.var(self.image1) > np.var(self.image2):
                self.image2 = match_histograms(self.image2, self.image1,
                                               multichannel=(not gray))
            else:
                self.image1 = match_histograms(self.image1, self.image2,
                                               multichannel=(not gray))

        self.image1 = torch.tensor(self.image1).float()
        self.image2 = torch.tensor(self.image2).float()

        if gray:
            self.image1 = self.image1.unsqueeze(dim=0)
            self.image2 = self.image2.unsqueeze(dim=0)

        self.get_split_points()
        img1, img2 = self.crop()
        return img1, img2, self.coords, self.image_size


class RegisterImage():
    def __init__(self, model, model_path, args, patch_size=512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model(args)
        self.model.load_state_dict(torch.load(model_path), strict=False)
        self.model.eval().to(self.device)
        self.split = SplitImage(patch_size)

    def recombine_flow(self, original_shape, flow_patches, coords, patch_size):
        """Recombine the flow with the given weight mask."""
        flow = torch.zeros((2, original_shape[0], original_shape[1]))

        # Patch generation
        weight = torch.zeros((original_shape[0], original_shape[1]))
        w_patch_mean = np.array([patch_size[0] / 2, patch_size[1] / 2])
        w_patch_cov = np.zeros((2, 2))
        w_patch_cov[0, 0] = 15 * patch_size[0]
        w_patch_cov[1, 1] = 15 * patch_size[1]
        X = np.linspace(0, patch_size[0] - 1, patch_size[0])
        Y = np.linspace(0, patch_size[1] - 1, patch_size[1])
        X, Y = np.meshgrid(X, Y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        weight_patch = multivar.pdf(pos, w_patch_mean, w_patch_cov)
        weight_patch = torch.from_numpy(weight_patch)

        # Recombine patches
        for i, coord in enumerate(coords):
            coord1 = coord[0]
            coord2 = coord[1]
            flow[:, coord1[0]: coord2[0], coord1[1]: coord2[1]] += flow_patches[i] * weight_patch
            weight[coord1[0]: coord2[0], coord1[1]: coord2[1]] += weight_patch
        return flow / weight

    def __call__(self, img_path1, img_path2, scale):
        """Load, split and register patches, then recombine the flow."""
        split_1, split_2, coords, image_size = self.split(img_path1, img_path2, scale)
        flows = list()
        for i in range(len(split_1)):
            self.model.eval()
            img1 = split_1[i].to(self.device).unsqueeze(dim=0)
            img2 = split_2[i].to(self.device).unsqueeze(dim=0)
            with torch.no_grad():
                flow_patch = self.model(img1, img2)[-1]
                flows.append(flow_patch.cpu().squeeze())
                del flow_patch
        flow = self.recombine_flow(image_size, flows, coords, self.split.patch_size)

        image_original = self.split.image1
        image_target = self.split.image2

        image_warp = warp(image_original, flow).squeeze()

        return image_original, image_target, image_warp, flow
