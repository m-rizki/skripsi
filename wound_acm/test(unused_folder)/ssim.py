import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.color import rgb2gray
from skimage.io import imread


kategori = "luka_hitam"
path = "dataset_3/"+ kategori +"/ready/"
path_skripsi = "D:/RL/skripsweet/dataset/dataset_3/"+ kategori +"/ready/"
img_name = "2"
extension = ".jpg"


groundtruth = rgb2gray(imread(path + img_name + "_r" + extension))
integer_region = rgb2gray(imread(path + img_name + "_integer_r" + extension))


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 5),
                         sharex=True, sharey=True)
ax = axes.ravel()

ssim_none = ssim(groundtruth, groundtruth, data_range=groundtruth.max() - groundtruth.min())

ssim_target = ssim(groundtruth, integer_region,
                  data_range=groundtruth.max() - groundtruth.min())

ax[0].imshow(groundtruth, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[0].set_xlabel(f'SSIM: {ssim_none:.2f}')
ax[0].set_title('Original image')

ax[1].imshow(integer_region, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_xlabel(f'SSIM: {ssim_target:.2f}')
ax[1].set_title('ssim_integer')

plt.tight_layout()
plt.show()