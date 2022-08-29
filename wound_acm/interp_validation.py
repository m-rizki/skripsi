import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
import matplotlib.pyplot as plt


path = "dataset_3/luka_merah/ready/"
img_name = "44"
extension = ".jpg"

groundtruth = rgb2gray(imread(path + img_name + "_r" + extension))
interp_region = rgb2gray(imread(path + img_name + "_interp_r" + extension))

sum_gt = 0
sum_interp = 0

for x in groundtruth:
    for y in x:
        if y != 1.0:
            sum_gt += 1

for p in interp_region:
    for q in p:
        if q != 1.0:
            sum_interp += 1

print(sum_gt)
print(abs(sum_gt-sum_interp))
print(abs(sum_gt-sum_interp) / sum_gt * 100)
print(100 - abs(sum_gt-sum_interp) / sum_gt * 100)
