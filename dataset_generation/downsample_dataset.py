import numpy as np
from PIL import Image
import cv2
import csv
from data_management import save_image, get_all_paths, load_csv, load_image


def downsample(image, new_small_side):
    """
    Downsample the image.

    Parameters
    ----------
    image : ndarray
        Image to be downsampled.
    new_small_side : int
        Size of the new smaller side.

    Returns
    -------
    image_d : ndarray
        Downsampled image with a pre-defined smaller side and the same ascpect ratio.
    """
    image_shape = image.shape[:2]
    ratio = image_shape[0] / image_shape[1]
    if ratio > 1:
        x = new_small_side
        y = np.round(new_small_side * ratio, 0).astype(int)
    else:
        y = new_small_side
        x = np.round(new_small_side / ratio, 0).astype(int)
    shape = (x, y)

    image_d = cv2.resize(image, shape, interpolation=cv2.INTER_LINEAR)

    return image_d

def interpolate_data(landmarks, shape_old, shape_new):
    """
    Scale landmarks to fit the new size.

    Parameters
    ----------
    landmarks : list of tuples
        Landmarks of the original image.
    shape_old : tuple
        Image size of the original image.
    shape_new : tuple
        Image size of the new image.

    Returns
    -------
    landmarks : list of tuples
        Landmarks scaled to fit the new image size.

    """
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


def save_landmarks(landmarks, filename):
    """
    Save landmarks.

    Parameters
    ----------
    landmarks : list of tuples
        List of landmarks.
    filename : string
        Save path.
    """
    with open(filename + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        header = ['idx', 'X', 'Y']
        writer.writerow(header)
        for i, landmark in enumerate(landmarks):
            row = [str(i), str(landmark[1]), str(landmark[0])]
            writer.writerow(row)


def generate_downsampled_(path, save_path, new_small_side):
    """
    Downsample the image, scale the landmarks and save both.

    Parameters
    ----------
    path : string
        Path to the original image and landmarks.
    save_path : string
        Save path.
    new_small_side : int
        Smaller side of the new image.

    """
    # Load
    image = load_image(path, extension='.png')
    landmarks = load_csv(path)

    # Downsample
    shape_old = image.shape[:2]
    image = downsample(image, new_small_side)
    shape_new = image.shape[:2]
    landmarks = interpolate_data(landmarks, shape_old, shape_new)

    # Save data
    save_image(image, save_path + '.png', const=1)
    save_landmarks(landmarks, save_path)
