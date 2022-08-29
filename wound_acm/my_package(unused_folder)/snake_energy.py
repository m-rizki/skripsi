import numpy as np
from skimage.filters import gaussian, sobel


# internal energy
def matrix(alpha, beta, gamma, len):
    
    # matrix
    a = beta
    b = -(4*beta + alpha)
    c = 6*beta + 2*alpha

    eye_n = np.eye(len, dtype=float)
    c_axis = c * eye_n
    b_axis = b * ( np.roll(eye_n, -1, axis=0) + np.roll(eye_n, -1, axis=1) )
    a_axis = a * ( np.roll(eye_n, -2, axis=0) + np.roll(eye_n, -2, axis=1) )
    A = c_axis + b_axis + a_axis

    # Only one inversion is needed for implicit spline energy minimization:
    inv = np.linalg.inv(A + gamma * eye_n)
    # can use float_dtype once we have computed the inverse in double precision
    inv = inv.astype(float, copy=False)
    
    return inv

# external energy
def external_gaussian(image, sigma, type="non_binary"):
    
    if type == "binary":
        external = gaussian(image, sigma)
        return external

    sobel_mag = sobel(gaussian(image, sigma))
    external = sobel_mag

    return external

def external(image, type="non_binary"):
    if type == "binary":
        return image
    
    sobel_mag = sobel(image)
    external = -(sobel_mag**2)
    return external


