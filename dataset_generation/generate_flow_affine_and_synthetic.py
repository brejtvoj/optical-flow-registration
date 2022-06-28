import numpy as np
import torch
import torch.nn.functional as F

from copy import copy
from data_management import load_image, load_csv

from generate_affine import generate_affine_flow, estimate_affine_parameters
from synthetic_flow import generate_flow as generate_flow_synthetic
from utils_warp import warp
from skimage.exposure.histogram_matching import match_histograms
from mask_creation import segment_wrapper


class FlowGenerator():
    """Class for generation of the training triplets."""

    def __init__(self, path):
        self.pairs = self.get_all_pairs(path)
        params = estimate_affine_parameters(self.pairs)
        self.affine_mean, self.affine_var = params
        self.image1 = None
        self.image2 = None
        self.landmarks = None
        self.flow = None

    def __call__(self, out_image_size=None, synthetic_md=None, affine_md=None,
                 use_mask=False):
        """
        Generate the training triplet.

        Parameters
        ----------
        out_image_size : tuple or ndarray
            Size of the images and the flow
        synthetic_md : float
            Maximum displacement of the synthetic flow.
        affine_md : float
            Maximum displacemnt of the affine transformation
        use_mask : bool
            Determin, whether the mask gets created.

        Returns
        -------
        image1 : ndarray
            Original image.
        image2 : ndarray
            Original image warped with generated flow and histogram matched.
        flow : ndarray
            Generated flow.
        mask : ndarray
            Foreground mask of the second image.
        """
        # Randomly select one of the pairs
        idx = np.random.randint(0, len(self.pairs) - 1)
        path1, path2 = self.pairs[idx]
        # Randomly switch the original and the target image
        if np.random.uniform(0, 1) > 0.5:
            path_c = copy(path1)
            path1 = path2
            path2 = path_c

        # Load images and landmarks
        self.landmarks = load_csv(path1)
        self.image1 = load_image(path1)
        self.image2 = load_image(path2)

        # Resize
        if out_image_size is not None:
            if isinstance(out_image_size, float) or isinstance(out_image_size, int):
                c = out_image_size
                y = int(self.image1.shape[0] * c)
                x = int(self.image1.shape[1] * c)
                out_image_size = (y, x)
            self.image1 = self.interpolate_image(self.image1, out_image_size)
            self.image2 = self.interpolate_image(self.image2, out_image_size)

        # Create a copy of the image1 and match its histograms to the image2
        self.image_copy = copy(self.image1)
        self.image_copy = match_histograms(self.image_copy, self.image2)
        self.image2 = self.image_copy

        # Generate affine portion of the flow
        M = np.eye(3)
        for i in range(3):
            for j in range(3):
                mean = self.affine_mean[i, j]
                var = self.affine_var[i, j]
                M[i, j] = np.random.uniform(mean - var, mean + var, 1)
        M[1, 0] = -M[0, 1] * np.random.uniform(0.975, 1.025)
        M[0, 2] *= self.image1.shape[0]
        M[1, 2] *= self.image2.shape[1]

        # Additional rotation
        theta = np.random.uniform(-np.pi / 4, np.pi / 4, 1)
        c = np.cos(theta)
        s = np.sin(theta)
        rotation_add = np.array([c, -s, s, c]).reshape((2, 2))
        M[0:2, 0:2] *= np.random.choice((-1, 1)) * rotation_add
        M = M if np.random.uniform(-1, 1) < 0 else np.linalg.inv(M)

        # Affine flow generation
        flow_affine = generate_affine_flow(self.image1.shape, M)
        max_affine_flow = np.max(np.abs(flow_affine))
        diag = np.sqrt(self.image1.shape[0]**2 + self.image2.shape[1]**2)

        # 0.92 and 0.15 chosen from experiments (ratio of affine to non-rigid)
        if not affine_md:
            affine_md = diag * 0.92 * 0.15

        flow_affine = flow_affine / max_affine_flow * affine_md

        # Generate random flow from multivariete normal distribution
        flow_shape = np.array(self.image1.shape[0:2])
        flow_synthetic = generate_flow_synthetic(flow_shape, count=20)

        # 0.08 and 0.15 chosen from experiments
        if not synthetic_md:
            synthetic_md = diag * 0.08 * 0.15

        flow_synthetic = flow_synthetic * synthetic_md

        # Generate combined flow and warp the image
        self.flow = flow_affine + flow_synthetic
        self.image2 = self.warp_image(self.image2, self.flow)

        if use_mask:
            mask = segment_wrapper(self.image2)
        else:
            mask = None

        return self.image1, self.image2, self.flow, mask

    def warp_image(self, image, flow):
        """
        Prepare and warp the image with optical flow.

        Parameters
        ----------
        image : ndarray
            Image to be warped.
        flow : ndarray
            Flow used for warping.

        Returns
        -------
        image_w : ndarray
            Warped image.
        """
        # Prepate data
        flow = torch.from_numpy(flow).unsqueeze(dim=0).float()
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(dim=0).float()

        image_w = warp(image, flow)

        image_w = image_w.squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
        return image_w

    def interpolate_image(self, image, size):
        """
        Prepare and interpolate the image.

        Parameters
        ----------
        image : ndarray
            Image to be interpolated.
        size : tuple or ndarray
            Size of the ouput image

        Returns
        -------
        image_i : ndarray
            Warped image.
        """
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(dim=0).float()

        image_i = F.interpolate(image, size)

        image_i = image_i.squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
        return image_i
