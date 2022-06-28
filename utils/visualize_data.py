import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


def plt_img_landmarks(image, landmarks):
    """
    Plot landmarks as red croees over the image.

    Parameters
    ----------
    image : ndarray
    landmarks : list of tuples
        Landmarks of the image.

    """
    plt.imshow(image)
    y = [landmarks[i][0] for i in range(len(landmarks))]
    x = [landmarks[i][1] for i in range(len(landmarks))]
    plt.scatter(x, y, c='r', marker='+', linewidths=0.5)
    plt.show()


def plt_img_diff(image1, image2, landmarks1, landmarks2,
                 path=None, reduced=None):
    """
    Plot images over each other with colored corresponding image pairs.

    Parameters
    ----------
    image1 : ndarray

    image2 : ndarray

    landmarks1 : list of tuples
        Landmarks of the first image.
    landmarks2 : list of tuples
        Landmarks of the second image.
    path : string
        Save path for the image.
    reduced : list of bools
        Determin, which color to use on the landmark pair.
    """
    # Preprocessing
    image1 = np.mean(image1, axis=2)**3
    image2 = np.mean(image2, axis=2)**2
    image1 = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
    image2 = (image2 - np.min(image2)) / (np.max(image2) - np.min(image2))

    # Plot image difference
    plt.imshow(image1, alpha=0.5, cmap='hot')
    plt.imshow(1-image2, alpha=0.5, cmap='bone')

    # Plot landmark difference
    for i in range(len(landmarks1)):
        y = (landmarks1[i][0], landmarks2[i][0])
        x = (landmarks1[i][1], landmarks2[i][1])
        if type(reduced) == ndarray:
            c = 'lime' if reduced[i] else 'red'
        else:
            c = 'blue'
        plt.plot(x, y, marker='+', linestyle='-.',
                 linewidth=0.3, markersize=1.6, color=c, alpha=1)

    # Figure settings
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.box(False)
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()
