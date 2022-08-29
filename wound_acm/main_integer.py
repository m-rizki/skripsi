import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian, sobel
from skimage.io import imread

my_dpi = 96 # https://www.infobyip.com/detectmonitordpi.php

# input data
kategori = "luka_merah"
path = "dataset_3/"+kategori+"/ready/"
path_skripsi = "D:/RL/skripsweet/dataset/dataset_3/"+ kategori +"/ready/"
img_name = "42"
extension = ".jpg"
img = imread(path + img_name + extension)
im_gray = rgb2gray(img)
im_gt = imread(path + img_name + "_g" + extension)


# parameter set
cr=160
cc=180
rad=90
sigma=3.5
sample=100
alpha = 1
beta = 10
gamma = 2 # time step
max_iter=100

# snake init (circle)
theta = np.linspace(0, 2*np.pi, sample)
r = cr + rad*np.sin(theta)
c = cc + rad*np.cos(theta)
snake_init = np.array([r, c]).T
# snake_init = np.around(snake_init).astype(int)

# snake_xy = snake_init[:, ::-1].astype(int)
# x = snake_xy[:, 0].astype(int)
# y = snake_xy[:, 1].astype(int)
# n = len(x)

snake_xy = snake_init[:, ::-1]
x = snake_xy[:, 0]
y = snake_xy[:, 1]
n = len(x)

fig0= plt.figure(frameon=False, figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
ax0 = fig0.add_axes([0, 0, 1, 1])
ax0.imshow(im_gray, cmap=plt.cm.gray)
ax0.plot(snake_xy[:, 0], snake_xy[:, 1], '-r', lw=2)
ax0.set_xticks([]), ax0.set_yticks([]) # hide axes
ax0.axis('off')
outname0 = path + img_name +"_integer_init"+ extension
outname0_skripsi = path_skripsi + img_name +"_integer_init"+ extension
plt.savefig(outname0, dpi=my_dpi)
plt.savefig(outname0_skripsi, dpi=my_dpi)
print(outname0 + " has been saved")
print(outname0_skripsi + " has been saved")
plt.close(fig0)


# energy external
ext = gaussian(im_gray, sigma)
ext = sobel(ext)
ext = -ext**2

# save energy external
fig1= plt.figure(frameon=False, figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
ax1 = fig1.add_axes([0, 0, 1, 1])
ax1.imshow(ext, cmap=plt.cm.gray)
ax1.set_xticks([]), ax1.set_yticks([]) # hide axes
ax1.axis('off')
outname1 = path + img_name +"_integer_ext"+ extension
outname1_skripsi = path_skripsi + img_name +"_integer_ext"+ extension
plt.savefig(outname1, dpi=my_dpi)
plt.savefig(outname1_skripsi, dpi=my_dpi)
print(outname1 + " has been saved")
print(outname1_skripsi + " has been saved")
plt.close(fig1)


# energy internal
# matrix
a = beta
b = -(4*beta + alpha)
c = 6*beta + 2*alpha

eye_n = np.eye(n, dtype=float)

c_axis = c * eye_n
b_axis = b * ( np.roll(eye_n, -1, axis=0) + np.roll(eye_n, -1, axis=1) )
a_axis = a * ( np.roll(eye_n, -2, axis=0) + np.roll(eye_n, -2, axis=1) )

A = c_axis + b_axis + a_axis
inv = np.linalg.inv(eye_n + gamma * A) # acton, ivins matrix equation

# potential force
gy, gx = np.gradient(ext)

# deform snake
fig2= plt.figure(frameon=False, figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
ax2 = fig2.add_axes([0, 0, 1, 1])
ax2.imshow(im_gray, cmap=plt.cm.gray)

xt = np.copy(x)
yt = np.copy(y)

for i in range(max_iter):
        fx = np.array([])
        fy = np.array([])
        
        for i in range(n):
                # fx = np.append(fx, gx[yt[i]] [xt[i]] )
                # fy = np.append(fy, gy[yt[i]] [xt[i]] )
                fx = np.append(fx, gx[np.round(yt[i]).astype(int)] [np.round(xt[i]).astype(int)] )
                fy = np.append(fy, gy[np.round(yt[i]).astype(int)] [np.round(xt[i]).astype(int)] )
        
        xn = np.dot(inv, xt + gamma * fx) # acton, ivins equation
        yn = np.dot(inv, yt + gamma * fy)

        # xt = np.round(xn).astype(int)
        # yt = np.round(yn).astype(int)
        xt = xn
        yt = yn

        # ax2.plot(xt, yt, '-g', lw=2)

snake_final = np.array([xt, yt]).T

ax2.plot(snake_xy[:, 0], snake_xy[:, 1], '-r', lw=2)
ax2.plot(snake_final[:, 0], snake_final[:, 1], '-b', lw=2)
ax2.set_xticks([]), ax2.set_yticks([]) # hide axes
ax2.axis('off')
outname2 = path + img_name +"_integer_result"+ extension
outname2_skripsi = path_skripsi + img_name +"_integer_result"+ extension
plt.savefig(outname2, dpi=my_dpi)
plt.savefig(outname2_skripsi, dpi=my_dpi)
print(outname2 + " has been saved")
print(outname2_skripsi + " has been saved")
# plt.show()
plt.close(fig2)

# save final contour region
fig3= plt.figure(frameon=False, figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
ax3 = fig3.add_axes([0, 0, 1, 1])
ax3.imshow(np.ones(img.shape))
ax3.fill(snake_final[:, 0], snake_final[:, 1], color="black")
ax3.set_xticks([]), ax3.set_yticks([]) # hide axes
ax3.axis('off')
outname3 = path + img_name +"_integer_r"+ extension
outname3_skripsi = path_skripsi + img_name +"_integer_r"+ extension
plt.savefig(outname3, dpi=my_dpi)
plt.savefig(outname3_skripsi, dpi=my_dpi)
print(outname3 + " has been saved")
print(outname3_skripsi + " has been saved")
plt.close(fig3)


# save overlay groundtruth & final snake

fig4= plt.figure(frameon=False, figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
ax4 = fig4.add_axes([0, 0, 1, 1])
ax4.imshow(im_gt, cmap=plt.cm.gray)
ax4.plot(snake_final[:, 0], snake_final[:, 1], '-b', lw=2)
ax4.set_xticks([]), ax4.set_yticks([]) # hide axes
ax4.axis('off')
outname4 = path + img_name +"_gt_r_integer"+ extension
outname4_skripsi = path_skripsi + img_name +"_gt_r_integer"+ extension
plt.savefig(outname4, dpi=my_dpi)
plt.savefig(outname4_skripsi, dpi=my_dpi)
print(outname4 + " has been saved")
print(outname4_skripsi + " has been saved")
plt.close(fig4)