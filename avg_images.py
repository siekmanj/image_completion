# Generates Fig. 6, the average composite image from MNIST and the three color channels of CIFAR10
import numpy as np
import random

from cifar10_web import cifar10
from mnist import MNIST

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os, sys, time, datetime

    mndata = MNIST('./data/mnist')
    mndata.gz = True
    mnist_images, labels = mndata.load_training()

    cifar_images, _, _, _ = cifar10(path='data/cifar')

    mnist_data   = [np.reshape(i, (28,28))            for i in mnist_images]
    cifar_r_data = [np.reshape(i[:1024],     (32,32)) for i in cifar_images]
    cifar_g_data = [np.reshape(i[1024:2048], (32,32)) for i in cifar_images]
    cifar_b_data = [np.reshape(i[2048:3072], (32,32)) for i in cifar_images]

    fig = plt.figure(figsize=(8,5))
    axs = []
    cols = 2
    rows = 3

    i = 1
    axs.append(fig.add_subplot(cols, rows, 2))
    plt.imshow(np.mean(mnist_data, axis=0), cmap='gray')

    axs.append(fig.add_subplot(cols, rows, 4))
    plt.imshow(np.mean(cifar_r_data, axis=0), cmap='gray')

    axs.append(fig.add_subplot(cols, rows, 5))
    plt.imshow(np.mean(cifar_g_data, axis=0), cmap='gray')

    axs.append(fig.add_subplot(cols, rows, 6))
    plt.imshow(np.mean(cifar_b_data, axis=0), cmap='gray')

    for ax in axs:
      ax.set_xticks([])
      ax.set_yticks([])
    plt.savefig('avg_images.png')
    plt.show()
