import numpy as np

import torch
import torch.nn.functional as F

from mask_creation import segment_wrapper as segment
from mask_creation import blur

from skimage.exposure.histogram_matching import match_histograms

from data_management import load_csv, load_image, get_paths
from copy import copy


class Datapoint():
    """
    Class for storing and generation of the training triplets.

    Flow is generated from landmarks.
    """

    def __init__(self, path1, path2):
        # Data loading
        self.data1 = load_csv(path1)
        self.data2 = load_csv(path2)
        # Reduce landmarks if shapes don't match
        if len(self.data1) != len(self.data2):
            max_len = min(len(self.data1), len(self.data2))
            self.data1 = self.data1[:max_len]
            self.data2 = self.data2[:max_len]

        self.num_data = len(self.data1)
        self.image = load_image(path1)
        self.image_t = load_image(path2)
        self.flow_shape = self.image.shape[:2]

        # Preprocessing
        self.interpolate_data()
        self.preprocess()

        # Flow generation
        self.flow_vectors = self.get_flow_vectors()
        self.flow = self.generate_flow_knn()

    def interpolate_data(self):
        """Interpolate images to be the same shape and transform landmarks accordingly."""
        # Save old shape
        shape_old = np.array(self.image_t.shape[:2])
        center_old = shape_old / 2

        # Interpolate the second image to be the same size as the first one
        self.image_t = torch.from_numpy(self.image_t).permute(2, 0, 1)
        self.image_t = self.image_t.unsqueeze(dim=0)
        self.image_t = F.interpolate(self.image_t, self.flow_shape)
        self.image_t = self.image_t.squeeze().permute(1, 2, 0).numpy()

        # Save new shape
        shape_new = np.array(self.image_t.shape[:2])
        center_new = shape_new / 2

        # Change in size
        y_coeff = shape_new[0] / shape_old[0]
        x_coeff = shape_new[1] / shape_old[1]

        # Scale landmarks to fit the new size
        for i in range(self.num_data):
            y_old = self.data2[i][0]
            x_old = self.data2[i][1]
            y_new = y_coeff * (y_old - center_old[0]) + center_new[0]
            x_new = x_coeff * (x_old - center_old[1]) + center_new[1]
            self.data2[i] = (y_new, x_new)

    def preprocess(self, grayscale=False, match=False):
        """
        Convert the image to grayscale and match histograms.

        Parameters
        ----------
        grayscale : bool
            Determin, whether to convert image to grayscale.
        match : bool
            Determin, whether to match histograms.
        """
        if grayscale:
            self.image = np.mean(self.image, axis=2)
            self.image_t = np.mean(self.image_t, axis=2)

        if match:
            if np.var(self.image) > np.var(self.image_t):
                self.image_t = match_histograms(self.image_t, self.image,
                                                multichannel=(not grayscale))
            else:
                self.image = match_histograms(self.image, self.image_t,
                                              multichannel=(not grayscale))

    def get_flow_vectors(self):
        """
        Calculate from in the landmarks.

        Returns
        -------
        vectors: list of tuples
            (y, x) movement of the landmarks.
        """
        vectors = list()

        for i in range(self.num_data):
            y_diff = self.data2[i][0] - self.data1[i][0]
            x_diff = self.data2[i][1] - self.data1[i][1]

            start = self.data1[i]

            direction = (y_diff, x_diff)

            vectors.append((start, direction))
        return vectors

    def generate_centers(self):
        """Convert list of centerns to a ndarray."""
        count = self.num_data
        centers = np.zeros((2, self.num_data))

        for i in range(count):
            centers[:, i] = self.flow_vectors[i][0]

        return centers

    def generate_flow_knn(self):
        """Generate flow from values in k-nearest neaighbours."""
        # Setup
        flow_shape = self.flow_shape
        centers = self.generate_centers()
        pdf_x = np.zeros((flow_shape[0], flow_shape[1]))
        pdf_y = np.zeros((flow_shape[0], flow_shape[1]))

        # Calculate flow for each pixel
        for y in range(flow_shape[0]):
            for x in range(flow_shape[1]):
                flow_y, flow_x = self.knn(centers, np.array([y, x]))
                pdf_y[y, x] = flow_y
                pdf_x[y, x] = flow_x

        # Blur the flow slightly
        pdf_x = torch.from_numpy(pdf_x).unsqueeze(dim=0).unsqueeze(dim=0)
        pdf_y = torch.from_numpy(pdf_y).unsqueeze(dim=0).unsqueeze(dim=0)
        pdf_x = blur(pdf_x, strenght=5, repeats=5, pad=7).cpu().numpy()
        pdf_y = blur(pdf_y, strenght=5, repeats=5, pad=7).cpu().numpy()

        # Combine the x and y components of the flow
        flow = np.zeros((2, flow_shape[0], flow_shape[1]))
        flow[0, :, :] = pdf_x
        flow[1, :, :] = pdf_y

        return flow

    def knn(self, centers, point, k=3):
        """Implement the knn algorithm for creation of the flow."""
        # Find k-NN
        distances = np.linalg.norm(centers.T - point, ord=2, axis=1)
        k_idx = np.argpartition(distances, k)[0:k]
        distances_k = distances[k_idx]

        # Manually make sure points around landmarks have the correct flow
        if np.any(distances_k**2 < 1e-2):
            flow_y = self.flow_vectors[k_idx[0]][1][0]
            flow_x = self.flow_vectors[k_idx[0]][1][1]
            return flow_y, flow_x

        # Weighted sum of values in k-NN
        flow_y = 0.
        flow_x = 0.
        w = np.sum((1 / distances_k)**2)
        for i in range(k):
            flow_y += self.flow_vectors[k_idx[i]][1][0] / distances[k_idx[i]]**2
            flow_x += self.flow_vectors[k_idx[i]][1][1] / distances[k_idx[i]]**2
        flow_y /= w
        flow_x /= w

        return flow_y, flow_x


def generate_training_data(path, create_mask=False):
    """
    Generate the training triplet and mask.

    Parameters
    ----------
    path : string
        Folder with the dataset.
    create_mask : bool
        Determin, whether the mask is created.

    Returns
    -------
    image1 : ndarray
        Original image.
    image2 : ndarray
        Target image.
    flow : ndarray
        Corresponding flow.
    mask : ndarray
        Foreground mask from the second image.
    """
    # Generate flow from landmarks
    path1, path2 = get_paths(path)
    data = Datapoint(path1, path2)
    image1 = copy(data.image)
    image2 = copy(data.image_t)
    flow = data.flow

    # Create a mask, roughly segmenting foreground and background
    if create_mask:
        mask = segment(image2)
    else:
        mask = None

    return image1, image2, flow, mask


def normalize_img(image):
    """
    Normalize image to the [0, 1] range.

    Parameters
    ----------
    image : ndarray
        Image to be normalized.

    Returns
    -------
    image_n: ndarray
        Image scaled to the [0, 1] range.
    """
    image = np.array(image)
    image_n = np.zeros_like(image)
    for c in range(image.shape[0]):
        image_c = image[c, :, :]
        image_c = (image_c - np.min(image_c)) / (np.max(image_c) - np.min(image_c))
        image_n[c, :, :] = image_c
    return image_n
