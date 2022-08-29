import numpy as np

def circle_init(cr, cc, radius, num):
    """
    Parameters
    ----------
    cr : center row
    cc : center column
    radius : circle radius
    num : number of samples to generate
    """
    theta = np.linspace(0, 2*np.pi, num)
    r = cr + radius*np.sin(theta)
    c = cc + radius*np.cos(theta)

    return np.array([r, c]).T

def manual_init(bin_img):
    r = np.array([])
    c = np.array([])
    return np.array([r, c]).T