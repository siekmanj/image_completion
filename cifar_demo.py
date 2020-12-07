import numpy as np
import random

from cifar10_web import cifar10
from mnist_demo import corrupt_image
from complete import complete_matrix

def corrupt_channels(R, G, B, proportion):
    """
    Mask out proportion of entries in matrix and replace with val.
    """
    R_X = R.copy()
    G_X = G.copy()
    B_X = B.copy()
    px_to_remove = int(proportion * len(R.flatten()))

    entries = [(i, j) for i in range(R_X.shape[0]) for j in range(R_X.shape[1])]

    np.random.shuffle(entries)

    omega = entries[px_to_remove:]

    entries= entries[:px_to_remove]

    max_entry = np.max(R)
    for (i, j) in entries:
        R_X[i,j] = 0
        G_X[i,j] = 0
        B_X[i,j] = 0

    return R_X, G_X, B_X, omega

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    images, _, _, _ = cifar10(path='data/cifar')

    np.random.seed(3)

    batch = list(range(len(images)))
    np.random.shuffle(batch)

    generate_cifar_examples = True
    if generate_cifar_examples:
        fig = plt.figure(figsize=(8,1))
        axs = []

        rows = 1

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

        plt.savefig('examples_cifar.png', transparent=True)
        plt.show()

    generate_fig4=True
    if True:
      fig = plt.figure(figsize=(8,5))
      axs = []
      cols = 3
      img = images[33]
      ms = [0.05, 0.1, 0.5]
      rows = len(ms)
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

        axs.append(fig.add_subplot(rows, cols, (i*cols)+1))
        if i == 0:
            axs[-1].set_title('Corrupted Image')
        axs[-1].set_xticks([])
        axs[-1].set_yticks([])
        axs[-1].set_ylabel('{:2d}% corrupted'.format(int(100*m)), rotation=0, labelpad=40)
        plt.imshow(rgb_bad)

        axs.append(fig.add_subplot(rows, cols, (i*cols)+2))
        if i == 0:
            axs[-1].set_title('Recovered Image')
        axs[-1].set_xticks([])
        axs[-1].set_yticks([])
        plt.imshow(rgb_rec)

        axs.append(fig.add_subplot(rows, cols, (i*cols)+3))
        if i == 0:
            axs[-1].set_title('Original Image')
        axs[-1].set_xticks([])
        axs[-1].set_yticks([])
        plt.imshow(rgb)
      plt.savefig('corruption_cifar.png', transparent=True)
      plt.show() 
