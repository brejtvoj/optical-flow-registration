import numpy as np
from scipy.stats import multivariate_normal as multivar
from scipy.ndimage import gaussian_filter


def generate_flow_(flow_shape, count=100):
    """
    Generate a 2D dispacement map for a single direction.

    Parameters
    ----------
    flow_shape : tuple or ndarray
        Size of the flow.
    count : int
        Number of Gaussians in the mixture

    Returns
    -------
    flow_ : ndarray
        2D displacement map.
    """
    # Parameters generation
    centers, covs, dirs = generate_normal_distributions(count, flow_shape)

    # Grid generation
    X = np.linspace(0, flow_shape[0] - 1, flow_shape[0])
    Y = np.linspace(0, flow_shape[1] - 1, flow_shape[1])
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    pdf = np.zeros(flow_shape).T

    for i in range(count):
        # Load parameters of the given Gaussian
        center = centers[:, i]
        cov = covs[:, :, i]
        d = dirs[:, i]

        pdf += multivar.pdf(pos, center, cov, allow_singular=True) * d[0]

    # Additional smoothing
    flow_ = gaussian_filter(pdf, np.max(flow_shape) / 5)
    return flow_


def generate_normal_distributions(count, flow_shape):
    """
    Generate centers, covarinace and direction of the Gaussians in the mixture.

    Parameters
    ----------
    count : int
        Number of gaussians in the mixture.
    flow_shape : int
        Size of the flow.

    Returns
    -------
    centers : ndarray
        (y, x) coordinates of the centers.
    covs : ndarray
        Covariance matrix for each of the Gaussians.
    d : ndarray
        {-1, 1} value to determin the direction for each of the Gaussians.
    """
    # Centers
    centers = (np.random.rand(2, count).T * flow_shape).T

    # Covariance
    covs = (np.random.rand(2, 2, count).T * flow_shape).T
    covs = np.clip(covs, flow_shape[1] / 4, flow_shape[1]) / 2

    for i in range(count):
        scale = np.random.rand()
        covs[:, :, i] = np.dot(covs[:, :, i], covs[:, :, i].T) * scale

    # Direction
    d = (np.random.rand(2, count).T * flow_shape).T
    d = np.where(np.random.rand(2, count) > 0.5, d, -d)

    return centers, covs, d


def normalize_flow(flow):
    """
    Normalize flow to the [-1, 1] range.

    Parameters
    ----------
    flow : ndarray
        Flow to be normalized.

    Returns
    -------
    flow_n: ndarray
        Flow scaled to the [-1, 1] range.
    """
    flow_n = np.zeros_like(flow)
    for c in range(flow.shape[0]):
        flow_c = flow[c, :, :]

        # [0, 1]
        flow_c = (flow_c - np.min(flow_c)) / (np.max(flow_c) - np.min(flow_c))

        # [-1, 1]
        flow_c = 2 * flow_c - 1

        flow_n[c, :, :] = flow_c
    return flow_n


def generate_flow(flow_shape, count):
    """
    Generate syntheic flow as a Gaussian mixture.

    Parameters
    ----------
    flow_shape : tuple or ndarray
        Shape of the flow.
    count : int
        Number of Gaussian in the mixture

    Returns
    -------
    flow : ndarray
        Displacement map for each the directions scaled to [-1, 1].
    """
    flow_1 = generate_flow_(flow_shape, count)
    flow_2 = generate_flow_(flow_shape, count)

    flow = np.zeros((2, flow_shape[0], flow_shape[1]))

    flow[0, :, :] = flow_1.T
    flow[1, :, :] = flow_2.T
    flow_n = normalize_flow(flow)
    return flow_n
