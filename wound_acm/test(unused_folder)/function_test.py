import numpy as np
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2gray
from skimage.filters import gaussian, sobel


import matplotlib.pyplot as plt





# snake init
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

# internal energy
def matrix(alpha, beta, gamma, len):
    
    a = beta
    b = -(4*beta + alpha)
    c = 6*beta + 2*alpha

    eye_n = np.eye(len, dtype=float)
    
    c_axis = c * eye_n
    b_axis = b * ( np.roll(eye_n, -1, axis=0) + np.roll(eye_n, -1, axis=1) )
    a_axis = a * ( np.roll(eye_n, -2, axis=0) + np.roll(eye_n, -2, axis=1) )
    A = c_axis + b_axis + a_axis

    inv = np.linalg.inv(A + gamma * eye_n)
    inv = inv.astype(float, copy=False)
    
    return inv

# external energy
def external_gaussian(image, sigma, type="non_binary"):
    
    if type == "binary":
        external = gaussian(image, sigma)
        return external

    sobel_mag = sobel(gaussian(image, sigma))
    external = -(sobel_mag**2)

    return external

def external(image, type="non_binary"):
    if type == "binary":
        return image
    
    sobel_mag = sobel(image)
    external = -(sobel_mag**2)
    return external



# snake_init = circle_init(120, 260, 100, 400)

img = imread("../dataset_3/luka_hitam/ready/2.jpg")
im_gray = rgb2gray(img)

# external
# external = external_gaussian(im_gray, 3.5, type="binary")
# external = external_gaussian(im_gray, 3.5, type="non_binary")
# external = external(im_gray, type="non_binary")
# external = external(im_gray, type="binary")


my_dpi = 96 # https://www.infobyip.com/detectmonitordpi.php
fig= plt.figure(frameon=False, figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(external, cmap=plt.cm.gray)

# ax.plot(snake_init[:, 1], snake_init[:, 0], '--r', lw=2)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
plt.show()