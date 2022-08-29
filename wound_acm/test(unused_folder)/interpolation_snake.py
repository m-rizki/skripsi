# float version, interpolation
# using skimage equation
# first order derivative using rectbivariatespline

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.util import img_as_float
from skimage.filters import gaussian, sobel
from skimage.io import imread
from scipy.interpolate import RectBivariateSpline
# skimage version 0.19.3

# input data
img = imread("dataset_3/luka_hitam/ready/37.jpg")
img = rgb2gray(img)
img_ori = img
img = gaussian(img, 3.5)

s = np.linspace(0, 2*np.pi, 400)
r = 110 + 95*np.sin(s)
c = 125 + 95*np.cos(s)
snake = np.array([r, c]).T

# edge map
img = img_as_float(img)
img = img.astype(float, copy=False)

edge = sobel(img)
img = edge

# Interpolate for smoothness: (***********************************************)
intp = RectBivariateSpline(np.arange(img.shape[1]),
                            np.arange(img.shape[0]),
                            img.T, kx=2, ky=2, s=0)


snake_xy = snake[:, ::-1]
x = snake_xy[:, 0].astype(float)
y = snake_xy[:, 1].astype(float)
n = len(x)

alpha = 0.015
beta = 10
gamma = 0.001 # time step

# matrix
a = beta
b = -(4*beta + alpha)
c = 6*beta + 2*alpha

eye_n = np.eye(n, dtype=float)
c_axis = c * eye_n
b_axis = b * ( np.roll(eye_n, -1, axis=0) + np.roll(eye_n, -1, axis=1) )
a_axis = a * ( np.roll(eye_n, -2, axis=0) + np.roll(eye_n, -2, axis=1) )
A = c_axis + b_axis + a_axis

# Only one inversion is needed for implicit spline energy minimization:
inv = np.linalg.inv(A + gamma * eye_n)
# can use float_dtype once we have computed the inverse in double precision
inv = inv.astype(float, copy=False)

max_num_iter=500
max_px_move=1.0


my_dpi = 96 # https://www.infobyip.com/detectmonitordpi.php
fig= plt.figure(frameon=False, figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
ax = fig.add_axes([0, 0, 1, 1])
# fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img_ori, cmap=plt.cm.gray)

# Snake energy minimization
xt = np.copy(x)
yt = np.copy(y)

# Explicit time stepping for image energy minimization:
for i in range(max_num_iter):
        # RectBivariateSpline always returns float64, so call astype here
        fx = intp(xt, yt, dx=1, grid=False).astype(float, copy=False)
        fy = intp(xt, yt, dy=1, grid=False).astype(float, copy=False)

        # #deform snake
        xn = np.dot(inv, gamma * xt + fx) # # skimage equation
        yn = np.dot(inv, gamma * yt + fy) #

        # Movements are capped to max_px_move per iteration:
        dx = max_px_move * np.tanh(xn - xt)
        dy = max_px_move * np.tanh(yn - yt)

        xt += dx
        yt += dy

        if i % 10 == 0:
                snake_iter = np.stack([yt, xt], axis=1)
                ax.plot(snake_iter[:, 1], snake_iter[:, 0], '-g', lw=2)

snake_final = np.stack([yt, xt], axis=1)

ax.plot(snake[:, 1], snake[:, 0], '-r', lw=2)
ax.plot(snake_final[:, 1], snake_final[:, 0], '-b', lw=2)
# ax.plot(snake[:, 1], snake[:, 0], '--r', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
plt.show()