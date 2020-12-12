# Generates Fig. 5, comparing the recovery of images from MNIST vs. CIFAR-10.

import numpy as np
import random
from run_mnist import corrupt_image, get_mnist
from run_cifar import corrupt_channels, get_cifar

from complete import complete_matrix

from cifar10_web import cifar10
from mnist import MNIST

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os, sys, time, datetime

    mnist_data = get_mnist()

    cifar_images = get_cifar()

    cifar_r_data = [np.reshape(i[:1024],     (32,32)) for i in cifar_images]
    cifar_g_data = [np.reshape(i[1024:2048], (32,32)) for i in cifar_images]
    cifar_b_data = [np.reshape(i[2048:3072], (32,32)) for i in cifar_images]

    fig = plt.figure(figsize=(14,6))
    axs = []
    cols = 7

    ms = [0.05, 0.15, 0.5]
    rows = len(ms)
    for i in range(rows):
        print("Doing row {} of {}".format(i+1,rows))
        m = ms[i]
        print(mnist_data.shape)
        img = mnist_data[65]
        corrupted, omega = corrupt_image(img, m)
        recovered        = np.round(complete_matrix(corrupted, omega))

        axs.append(fig.add_subplot(rows, cols, (i*cols)+1))
        if i == 0:
            axs[-1].set_title('Corrupted Image')
        axs[-1].set_xticks([])
        axs[-1].set_yticks([])
        axs[-1].set_ylabel('{:2d}% corrupted'.format(int(100*m)), rotation=0, labelpad=40)
        plt.imshow(corrupted, cmap='gray')

        axs.append(fig.add_subplot(rows, cols, (i*cols)+2))
        if i == 0:
            axs[-1].set_title('Recovered Image')
        axs[-1].set_xticks([])
        axs[-1].set_yticks([])
        plt.imshow(recovered, cmap='gray')

        axs.append(fig.add_subplot(rows, cols, (i*cols)+3))
        if i == 0:
            axs[-1].set_title('Original Image')
        axs[-1].set_xticks([])
        axs[-1].set_yticks([])
        plt.imshow(img, cmap='gray')

        r = cifar_r_data[4]
        g = cifar_g_data[4]
        b = cifar_b_data[4]
        r_bad, g_bad, b_bad, omega = corrupt_channels(r,g,b,m)
        r_hat = complete_matrix(r_bad, omega)
        g_hat = complete_matrix(g_bad, omega)
        b_hat = complete_matrix(b_bad, omega)

        rgb     = np.dstack((r,g,b))
        rgb_bad = np.dstack((r_bad,g_bad,b_bad))
        rgb_rec = np.dstack((r_hat,g_hat,b_hat))

        axs.append(fig.add_subplot(rows, cols, (i*cols)+4))
        if i == 0:
            axs[-1].set_title('Corrupted Image')
        axs[-1].set_xticks([])
        axs[-1].set_yticks([])
        #axs[-1].set_ylabel('{:2d}% corrupted'.format(int(100*m)), rotation=0, labelpad=40)
        plt.imshow(rgb_bad)

        axs.append(fig.add_subplot(rows, cols, (i*cols)+5))
        if i == 0:
            axs[-1].set_title('Recovered Image')
        axs[-1].set_xticks([])
        axs[-1].set_yticks([])
        plt.imshow(rgb_rec)

        axs.append(fig.add_subplot(rows, cols, (i*cols)+6))
        if i == 0:
            axs[-1].set_title('Original Image')
        axs[-1].set_xticks([])
        axs[-1].set_yticks([])
        plt.imshow(rgb)


    plt.savefig('figs/fig_cmp.png')
    plt.show()
