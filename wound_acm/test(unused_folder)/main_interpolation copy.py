# import numpy as np
# from skimage.io import imread
# from skimage.util import img_as_float
# from skimage.color import rgb2gray
# from scipy.interpolate import interp1d, interp2d, RectBivariateSpline

# # from skimage.filters import gaussian, sobel

# import matplotlib.pyplot as plt

# # my_package
# from my_package.snake_init import circle_init
# from my_package.snake_energy import external, external_gaussian, matrix

# # snake init return vector array (y, x). snake_xy return vector array (x,y)
# cr = 120
# cc = 260
# radius = 100
# num = 400
# snake_init = circle_init(cr, cc, radius, num )

# snake_xy = snake_init[:, ::-1]
# x = snake_xy[:, 0].astype(float)
# y = snake_xy[:, 1].astype(float)


# # external energy return image
# img = imread("dataset_3/luka_hitam/ready/2.jpg")
# im_gray = rgb2gray(img)

# sigma = 3.5
# ext = external_gaussian(im_gray, sigma, type="non_binary")
# ext = img_as_float(ext)
# ext = ext.astype(float, copy=False)


# # internal energy return matrix inv
# alpha = 0.015
# beta = 10
# gamma = 0.001 # time step
# len = num
# inv = matrix(alpha, beta, gamma, len)

# # edge map
# gy, gx = np.gradient(ext)


# # Interpolate for smoothness
# intp_x = RectBivariateSpline(np.arange(gx.shape[1]),
#                             np.arange(gx.shape[0]),
#                             gx.T, kx=2, ky=2, s=0)

# intp_y = RectBivariateSpline(np.arange(gx.shape[1]),
#                             np.arange(gx.shape[0]),
#                             gy.T, kx=2, ky=2, s=0)

# max_num_iter = 800
# max_px_move = 1.0

# my_dpi = 96 # https://www.infobyip.com/detectmonitordpi.php
# fig= plt.figure(frameon=False, figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
# ax = fig.add_axes([0, 0, 1, 1])
# ax.imshow(img, cmap=plt.cm.gray)

# # Snake energy minimization
# xt = np.copy(x)
# yt = np.copy(y)
# for i in range(max_num_iter):
#         # RectBivariateSpline always returns float64, so call astype here (*******************************************)
#         fx = intp_x(xt, y, dx=0, grid=False).astype(float, copy=False)
#         fy = intp_y(xt, y, dy=0, grid=False).astype(float, copy=False)
        
#         # my_equation
#         xn = np.dot(inv, gamma * x - fx)
#         yn = np.dot(inv, gamma * y - fy)
       
#         # Movements are capped to max_px_move per iteration -> skimage
#         dx = max_px_move * np.tanh(xn - x)
#         dy = max_px_move * np.tanh(yn - y)
        
#         x += dx # (*******************************************)
#         y += dy #

#         if i % 10 == 0:
#                 snake_iter = np.stack([y, x], axis=1)
#                 ax.plot(snake_iter[:, 1], snake_iter[:, 0], '-g', lw=2)

# snake_final = np.stack([x, y], axis=1)

# ax.plot(snake_init[:, 1], snake_init[:, 0], '-r', lw=2)
# ax.plot(snake_final[:, 0], snake_final[:, 1], '-b', lw=2)
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])
# plt.show()




# # my_dpi = 96 # https://www.infobyip.com/detectmonitordpi.php
# # fig= plt.figure(frameon=False, figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
# # ax = fig.add_axes([0, 0, 1, 1])
# # ax.imshow(ext, cmap=plt.cm.gray)

# # ax.plot(snake_init[:, 1], snake_init[:, 0], '--r', lw=2)
# # ax.set_xticks([]), ax.set_yticks([])
# # ax.axis([0, ext.shape[1], ext.shape[0], 0])
# # plt.show()