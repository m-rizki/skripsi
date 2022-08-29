import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

kategori="luka_hitam"
path = "dataset_3/"+kategori+"/ready/"
img_name = "26_kurva_manual"
extension = ".jpg"
img = imread(path + img_name + extension)
img_gray = rgb2gray(img)

r , c = np.where(img_gray == 0)

# cr = 160
# cc = 250


snake_rc = np.array([r, c]).T
r = snake_rc[:, 1]
c = snake_rc[:, 0]
n = len(r)


# bucket algorithm
# nrows, ncols = img_gray.shape

# b1_r = np.array([])
# b1_c = np.array([])
# b2_r = np.array([])
# b2_r = np.array([])

# for i in range(nrows):
#     for j in range(ncols):
#         if img_gray == 0:
#             if i
#             print('')


# sort given set of Cartesian points based on polar angles
# import math
# def solve(points):
#    def key(x):
#       atan = math.atan2(x[1], x[0])
#       return (atan, x[1]**2+x[0]**2) if atan >= 0 else (2*math.pi + atan, x[0]**2+x[1]**2)

#    return sorted(points, key=key)

# points = [(1,1), (1,-2),(-2,2),(5,4),(4,5),(2,3),(-3,4)]
# print(solve(points))



fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(np.ones(img_gray.shape), cmap="gray", vmin=0, vmax=1)
ax.plot(snake_rc[:, 1], snake_rc[:, 0], "-r", lw=2)
# ax.plot(cc, cr, marker='o')
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img_gray.shape[1], img_gray.shape[0], 0])
plt.show()