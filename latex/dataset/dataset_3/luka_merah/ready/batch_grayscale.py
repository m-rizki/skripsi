import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray

data = ["1", "2", "3", "8", "9", "11", "14", "18", "19", "20", "22", "44"]

my_dpi = 96 # https://www.infobyip.com/detectmonitordpi.php

for i in data:
    
    filename = str(i)

    img = imread(filename + ".jpg")
    img = rgb2gray(img)

    # plot grayscale
    fig= plt.figure(frameon=False, figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap=plt.cm.gray)
    ax.set_xticks([]), ax.set_yticks([]) # hide axes
    ax.axis('off')
    
    outname = filename + "_gray" + ".jpg"

    plt.savefig(outname, dpi=my_dpi)
    print(outname + " has been saved")
    plt.close(fig)
