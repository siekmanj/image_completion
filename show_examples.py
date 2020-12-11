# Generates Figs. 1 and 2

from run_mnist import get_mnist
from run_cifar import get_cifar
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    Ms = get_mnist()[:5]
    fig = plt.figure(figsize=(8,1))
    axs = []

    rows = 1

    cols=5
    for i in range(rows*cols):
        axs.append(fig.add_subplot(rows, cols, i+1))
        axs[-1].set_xticks([])
        axs[-1].set_yticks([])
        plt.imshow(Ms[i], cmap='gray')
    plt.show()

    fig = plt.figure(figsize=(8,1))
    axs = []
    rows = 1
    images = get_cifar()
    batch = list(range(len(images)))
    np.random.shuffle(batch)
    cols=5
    for i in range(rows*cols):
        img = images[batch[i]]
        r = [img[j] for j in range(1024)]
        g = [img[j] for j in range(1024, 2048)]
        b = [img[j] for j in range(2048, 3072)]
        r = np.reshape(r, (32,32))
        g = np.reshape(g, (32,32))
        b = np.reshape(b, (32,32))
        rgb= np.dstack((r,g,b))
        axs.append(fig.add_subplot(rows, cols, i+1))
        axs[-1].set_xticks([])
        axs[-1].set_yticks([])
        plt.imshow(rgb)
    plt.show()
