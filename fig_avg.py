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
      #axs.set_ylabel(
    plt.savefig('avg_images.png')
    plt.show()
    """"
    for i, m in enumerate(ms):
      r = [img[j] for j in range(1024)]
      g = [img[j] for j in range(1024, 2048)]
      b = [img[j] for j in range(2048, 3072)]
      r = np.reshape(r, (32,32))
      g = np.reshape(g, (32,32))
      b = np.reshape(b, (32,32))

      r_bad, g_bad, b_bad, omega = corrupt_channels(r,g,b,m)
      r_hat = complete_matrix(r_bad, omega)
      g_hat = complete_matrix(g_bad, omega)
      b_hat = complete_matrix(b_bad, omega)

      rgb     = np.dstack((r,g,b))
      rgb_bad = np.dstack((r_bad,g_bad,b_bad))
      rgb_rec = np.dstack((r_hat,g_hat,b_hat))

      axs.append(fig.add_subplot(cols, rows, (i*rows)+1))
      if i == 0:
          axs[-1].set_title('Corrupted Image')
      axs[-1].set_xticks([])
      axs[-1].set_yticks([])
      axs[-1].set_ylabel('{:2d}% corrupted'.format(int(100*m)), rotation=0, labelpad=40)
      plt.imshow(rgb_bad)

      axs.append(fig.add_subplot(cols, rows, (i*rows)+2))
      if i == 0:
          axs[-1].set_title('Recovered Image')
      axs[-1].set_xticks([])
      axs[-1].set_yticks([])
      plt.imshow(rgb_rec)

      axs.append(fig.add_subplot(cols, rows, (i*rows)+3))
      if i == 0:
          axs[-1].set_title('Original Image')
      axs[-1].set_xticks([])
      axs[-1].set_yticks([])
      plt.imshow(rgb)
    plt.savefig('corruption_cifar.png', transparent=True)
    plt.show() 
    """

