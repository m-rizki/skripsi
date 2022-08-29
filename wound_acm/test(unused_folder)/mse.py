import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.color import rgb2gray
from skimage.io import imread



def mse(img1, img2):
	err = np.sum((img1.astype(float) - img1.astype(float)) ** 2)
	err /= float(img1.shape[0] * img1.shape[1])
	return err


kategori = "luka_hitam"
path = "dataset_3/"+ kategori +"/ready/"
path_skripsi = "D:/RL/skripsweet/dataset/dataset_3/"+ kategori +"/ready/"
img_name = "2"
extension = ".jpg"


groundtruth = rgb2gray(imread(path + img_name + "_r" + extension))
integer_region = rgb2gray(imread(path + img_name + "_integer_r" + extension))


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()

mse_none = mse(groundtruth, groundtruth)

mse_target = mse(groundtruth, integer_region)

ax[0].imshow(groundtruth, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[0].set_xlabel(f'MSE: {mse_none:.2f}')
ax[0].set_title('Original image')

ax[1].imshow(integer_region, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_xlabel(f'MSE: {mse_target:.2f}')
ax[1].set_title('mse_integer')

plt.tight_layout()
plt.show()
outname0 = path + img_name +"_integer_mse"+ extension
outname0_skripsi = path_skripsi + img_name +"_integer_mse"+ extension
plt.savefig(outname0)
plt.savefig(outname0_skripsi)
print(outname0 + " has been saved")
print(outname0_skripsi + " has been saved")
plt.close(fig)