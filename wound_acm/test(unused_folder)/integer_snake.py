# skimage equation
# integer version
# first order derivative using np.gradient

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.util import img_as_float
from skimage.filters import gaussian, sobel
from skimage.io import imread
from scipy.interpolate import RectBivariateSpline

# input data
img = imread("dataset_3/luka_hitam/ready/37.jpg")
img_gray = rgb2gray(img)

sigma = 3.5
img_gaussian = gaussian(img_gray, sigma)

# snake init
s = np.linspace(0, 2*np.pi, 200)
r = 110 + 95*np.sin(s)
c = 125 + 95*np.cos(s)
snake_init = np.array([r, c]).T


# round snake init
snake_init_int = np.around(snake_init).astype(int)

# snake xy (int)
snake_xy_int = snake_init_int[:, ::-1].astype(int)
x_int = snake_xy_int[:, 0].astype(int)
y_int = snake_xy_int[:, 1].astype(int)

# gradient magnitude square
img_sobel = sobel(img_gaussian)
gm_square = img_sobel**2
# negative gradient magnitude square
neg_gm_square = -gm_square

alpha = 1
beta = 10
dt = 6 # time step

n = snake_init.shape[0]

# matrix
a = beta
b = -(4*beta + alpha)
c = 6*beta + 2*alpha

eye_n = np.eye(n, dtype=float)

c_axis = c * eye_n
b_axis = b * ( np.roll(eye_n, -1, axis=0) + np.roll(eye_n, -1, axis=1) )
a_axis = a * ( np.roll(eye_n, -2, axis=0) + np.roll(eye_n, -2, axis=1) )

A = c_axis + b_axis + a_axis
inv = np.linalg.inv(eye_n + dt * A) # acton, ivins matrix equation

gy, gx = np.gradient(neg_gm_square)
# snake evolution
x_t = np.copy(x_int)
y_t = np.copy(y_int)

my_dpi = 96 # https://www.infobyip.com/detectmonitordpi.php
fig= plt.figure(frameon=False, figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
# fig, ax = plt.subplots()
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(img_gray, cmap=plt.cm.gray)
ax.plot(snake_init_int[:, 1], snake_init_int[:, 0], color='red', marker='o', linewidth=0, markersize=3)

max_iter = 100
for i in range(max_iter):
        fx = np.array([])
        fy = np.array([])
        
        for i in range(n):
                fx = np.append(fx, gx[y_t[i]] [x_t[i]] )
                fy = np.append(fy, gy[y_t[i]] [x_t[i]] )

        xn = np.dot(inv, x_t + dt * fx) # acton, ivins equation
        yn = np.dot(inv, y_t + dt * fy)

        x_t = np.round(xn).astype(int)
        y_t = np.round(yn).astype(int)

        ax.plot(x_t, y_t, color='yellow', marker='o', linewidth=0, markersize=3)

snake_final = np.array([y_t, x_t]).T
ax.plot(snake_final[:, 1], snake_final[:, 0], color='blue', marker='o', linewidth=0, markersize=3)
plt.show()