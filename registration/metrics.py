import numpy as np


def calc_measure(landmarks1, landmarks2, image_shape):
    """
    Calculate basic measures.

    Parameters
    ----------
    landmarks1 : list of tuples
        Landmarks of the first image.
    landmarks2 : list of tuples
        Landmarks of the second image.
    image_shape : tuple
        Size of the iamge.

    Returns
    -------
    Median EPE and median rTRE : float
    """
    if isinstance(landmarks1, list):
        landmarks1 = np.array(landmarks1)
    if isinstance(landmarks2, list):
        landmarks2 = np.array(landmarks2)
    EPE = np.linalg.norm(landmarks1 - landmarks2, ord=2, axis=1)
    rTRE = EPE / np.sqrt(image_shape[0]**2 + image_shape[1]**2)
    return np.median(EPE), np.median(rTRE)


def get_MrTRE(landmarks_all, shapes):
    """
    Calculate EPE and MrTRE for all landmarks.

    Parameters
    ----------
    landmarks_all : list of lists of tuples
        Landmark pairs of all the images.
    shapes : list of tuples
        Shapes of all the image pairs.

    Returns
    -------
    EPE_all : list of floats
        EPE value for all landmarks.
    MrTRE_all : list of floats
        MrTRE value for all landmarks.

    """
    n_pairs = len(landmarks_all)
    MrTRE_all = np.zeros(n_pairs)
    EPE_all = np.zeros(n_pairs)
    for i in range(n_pairs):
        l1, l2 = landmarks_all[i]
        EPE_all[i], MrTRE_all[i] = calc_measure(l1, l2, shapes[i][0])
    return EPE_all, MrTRE_all


def robustness(l_org, l_new):
    """
    Calculate robustness of the registration on the given image pair.

    Parameters
    ----------
    l_org : list of tuples
        Original image pairs.
    l_new : list of tuples
        Registered image pairs.

    Returns
    -------
    r : float
        Robusteness between the landmarks.

    """
    l_org_1, l_org_2 = [np.array(x) for x in l_org]
    l_new_1, l_new_2 = [np.array(x) for x in l_new]
    EPE_org = np.linalg.norm(l_org_1 - l_org_2, ord=2, axis=1)
    EPE_new = np.linalg.norm(l_new_1 - l_new_2, ord=2, axis=1)
    r = np.sum(EPE_org > EPE_new) / EPE_org.shape
    return r


def robustness_all(l_org_all, l_new_all):
    """
    Calculate robusteness for all the images.

    Parameters
    ----------
    l_org : tuple of lists
        Original image pairs.
    l_new : tuple of lists
        Registered image pairs.

    Returns
    -------
    AR and MR of the method.
    """
    r_all = np.zeros(len(l_org_all))
    for i in range(len(l_org_all)):
        r_all[i] = robustness(l_org_all[i], l_new_all[i])
    return np.mean(r_all), np.median(r_all)


def max_rTRE(lm1, lm2, shape):
    """
    Calculate MxrTRE for the given image pair.

    Parameters
    ----------
    lm1 : list of tuples
        Landmarks of the first image.
    lm2 : list of tuples
        Landmarks of the second image.
    shape : tuple
        Size of the image.

    Returns
    -------
    AMxrTRE and MMxrTRE : float

    """
    lm1 = np.array(lm1)
    lm2 = np.array(lm2)
    d = np.sqrt(shape[0]**2 + shape[1]**2)
    rTRE = np.linalg.norm(lm1 - lm2, ord=2, axis=1) / d
    return np.max(rTRE)


def max_rTRE_all(lm, shapes):
    """
    Calculate MxrTRE for all the image pairs.

    Parameters
    ----------
    lm : list of lists of tuples
        Complete set of all the landmarks.
    shapes : TYPE
        Complete set of all the image shapes.

    Returns
    -------
    AMxrTRE : float

    MMxrTRE : float
    """
    MxrTRE = np.zeros(len(lm))
    for i in range(len(lm)):
        MxrTRE[i] = max_rTRE(lm[i][0], lm[i][1], shapes[i][2])
    AMxrTRE = np.mean(MxrTRE)
    MMxrTRE = np.median(MxrTRE)
    return AMxrTRE, MMxrTRE

def rTRE_based(lm1, lm2, shape):
    """
    Calculate rTRE based metric for the image pair

    Parameters
    ----------
    lm1 : list of tuples
        Landmarks of the first image.
    lm2 : list of tuples
        Landmarks of the first image.
    shape : tuple
        Size of the image.

    Returns
    -------
    ArTRE and MrTRE : float
    """
    lm1 = np.array(lm1)
    lm2 = np.array(lm2)
    rTRE = np.linalg.norm(lm1 - lm2, ord=2, axis=1) / np.sqrt(shape[0]**2 + shape[1]**2)
    return np.mean(rTRE), np.median(rTRE)


def rTRE_based_all(lm, shapes):
    """
    Calculate all rTRE based metric for the entire dataset

    Parameters
    ----------
    lm : list of lists of tuples
        Landmarks of the entire dataset.
    shapes : list of tuples
        Shapes of all the images.

    Returns
    -------
    AArTRE : float

    MArTRE : float

    AMrTRE : float

    MMrTRE : float

    ArTRE : float
    """
    MrTRE = np.zeros(len(lm))
    ArTRE = np.zeros(len(lm))
    for i in range(len(lm)):
        ArTRE[i], MrTRE[i] = rTRE_based(lm[i][0], lm[i][1], shapes[i][2])
    AArTRE = np.mean(ArTRE)
    MArTRE = np.median(ArTRE)
    AMrTRE = np.mean(MrTRE)
    MMrTRE = np.median(MrTRE)
    return AArTRE, MArTRE, AMrTRE, MMrTRE, ArTRE
