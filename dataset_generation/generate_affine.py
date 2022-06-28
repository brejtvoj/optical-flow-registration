import numpy as np
import os
from data_management import load_image_and_landmarks


def generate_A(landmarks):
    """
    Generate matrix A from the equation AX = b.

    Each 2 rows are generated as:
        [xi, yi, 1, 0, 0, 0]
        [0, 0, 0, xi, yi, 1]

    Parameters
    ----------
    landmarks : ndarray
        Landmarks from the first image.

    Returns
    -------
    A : ndarray
        Matrix in the shape for least-squares fitting.
    """
    A = np.zeros((2 * landmarks.shape[0], 6))

    for i in range(landmarks.shape[0]):
        xi = landmarks[i, 0]
        yi = landmarks[i, 1]

        A[2 * i, :] = np.array([xi, yi, 1, 0, 0, 0])
        A[2 * i + 1, :] = np.array([0, 0, 0, xi, yi, 1])

    return A


def generate_b(landmarks):
    """
    Generate column vector b from the equation AX = b.

    Parameters
    ----------
    landmarks : ndarray
        Landmarks from the second image.

    Returns
    -------
    b : ndarray
        Vector in the shape for least-squares fitting.
    """
    b = landmarks.flatten()

    return b


def generate_X(A, b):
    """
    Solve for the matrix X, in the equation AX = b.

    Parameters
    ----------
    A : ndarray
        Landmarks from the first image, represented as a matrix.
    b : ndarray
        Landmarks from the second image, represented as a column vector.

    Returns
    -------
    X : ndarray
        Matrix of the affine transformation, which solves the equation AX = b,
        it the sense of least-squares.
    """
    X = np.eye(3)

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    X[:2, :] = np.reshape(x, (2, 3))
    return X


def find_affine(landmarks1, landmarks2):
    """
    Find the affine transformation between two sets of landmarks.

    Parameters
    ----------
    landmarks1 : ndarray
        Landmarks from the first image.
    landmarks2 : ndarray
        Landmarks from the second image.

    Returns
    -------
    X : ndarray
        Matrix of the affine transformation, which approximates the affine
        transformation from landmarks1 to landmarks2.
    """
    A = generate_A(landmarks1)
    b = generate_b(landmarks2)

    X = generate_X(A, b)

    return X


def transform_landmarks(landmarks, matrix):
    """
    Transform a set of landmarks with the given affine matrix.

    Parameters
    ----------
    landmarks : ndarray
        Landmarks from a image.
    matrix : ndarray
        Affine transformation matrix.

    Returns
    -------
    landmarks_out : ndarray
        Transformed landmarks.

    """
    landmarks_out = np.zeros_like(landmarks)

    for i in range(landmarks.shape[0]):
        lm = np.ones(3)
        lm[:2] = landmarks[i, :]

        landmarks_out[i, :] = np.dot(matrix, lm)[:2]

    return landmarks_out


def generate_affine_flow(image_shape, M):
    """
    Generate flow, which correspond to the given affine transformation.

    Parameters
    ----------
    image_shape: tuple or ndarray
        Shape of the image, for which we want to generate the flow.
    M : ndarray
        Affine transformation matrix.

    Returns
    -------
    flow : ndarray
        Flow generated from the affine transformation.
    """
    flow = np.zeros((2, image_shape[0], image_shape[1]))

    y = np.arange(image_shape[0]).astype(float)
    x = np.arange(image_shape[1]).astype(float)
    X, Y = np.meshgrid(x, y)

    Xt = M[0, 0] * X + M[0, 1] * Y + M[0, 2]
    Yt = M[1, 0] * X + M[1, 1] * Y + M[1, 2]

    flow[0, :, :] = Xt - X
    flow[1, :, :] = Yt - Y

    return flow


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


def estimate_affine_parameters(paths, verbose=False):
    """
    Estimate the affine transformation for all image pairs.

    Parameters
    ----------
    paths : string
        List of paths, for which we want to estimate the affine transformation.
    verbose : bool
        Print progress and results.

    Returns
    -------
    A_mean : ndarray
        Mean values of the affine transformation parameters.
    A_var : ndarray
        Variance of the affine transformation parameters.
    """
    A = np.zeros((len(paths), 3, 3))

    for i, path, in enumerate(paths):
        if verbose:
            print(f'Estimating affine parameters {i + 1} / {len(paths)}')

        path1, path2 = path
        image1, landmarks1 = load_image_and_landmarks(path1)
        image2, landmarks2 = load_image_and_landmarks(path2)

        # Reduce the amount of landmarks to be the same
        max_len = min(len(landmarks1), len(landmarks2))
        landmarks1 = landmarks1[:max_len, :]
        landmarks2 = landmarks2[:max_len, :]

        AT = find_affine(landmarks1, landmarks2)

        # Scale transaltion by the image diagonal
        AT[0:2, 2] /= np.sqrt(image1.shape[0]**2 + image1.shape[1]**2)

        A[i, :, :] = AT

    if verbose:
        print('Parameters estimated...')

    A_mean = np.mean(A, axis=0)
    A_var = np.var(A, axis=0)

    return A_mean, A_var
