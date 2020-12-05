import numpy as np
import random

from complete import complete_matrix, mask_out_matrix
from mnist import MNIST

def corrupt_image(M, proportion):
    """
    Mask out proportion of entries in matrix and replace with val.
    """
    X = M.copy()
    px_to_remove = int(proportion * len(X.flatten()))

    entries = [(i, j) for i in range(X.shape[0]) for j in range(X.shape[1])]

    np.random.shuffle(entries)

    omega = entries[px_to_remove:]

    entries= entries[:px_to_remove]

    max_entry = np.max(M)
    for (i, j) in entries:
        X[i,j] = max_entry**2

    return X, omega

def recovery_scatterplot(batch, ms=[0.01, 0.05]):
    data = {'idx': [], 'rank': []}

    for m in ms:
      data[str(m)] = []

    for i, img in enumerate(batch): 
      M            = np.reshape(img, (28,28))
      rank_m       = np.linalg.matrix_rank(M)
      data['idx']  += [i]
      data['rank'] += [rank_m]

      for m in ms:
        corrupted, omega = corrupt_image(M, m) # corrupt image by altering 100*m percent of pixels
        if debug:
          print('m:', m)
          print(MNIST.display((M).flatten()))

        recovered       = np.round(complete_matrix(corrupted, omega))

        frobenius_diff  = np.linalg.norm(M, 'nuc') - np.linalg.norm(recovered, 'nuc')
        err_frobenius   = np.linalg.norm(M - recovered, 'fro')
        recovery_metric = err_frobenius / np.linalg.norm(M, 'fro')

        data[str(m)] += [recovery_metric]

        if debug:
          pass
          print("recovered:")
          print(MNIST.display(recovered.flatten()))
        print("Original matrix rank: {:2d} | recovery {:6.5f}".format(rank_m, recovery_metric))

    #np.savez('rank_scatter.npz', xs=xs, ys=ys, cs=cs)
    fig, axs = plt.subplots(1,len(ms), figsize=(5*len(ms),4))
    for i, m in enumerate(ms):
        axs[i].set_title('m='+str(1-m))
        axs[i].plot(np.mean(data['rank'], axis=0), np.mean(data[str(m)], axis=0), s=2)
        
    plt.savefig('backup.png')
    plt.show()

if __name__ == '__main__':
    import os, time
    import matplotlib.pyplot as plt

    debug = False

    if not os.path.isfile('./data/mnist/train-images-idx3-ubyte.gz'):
      import urllib.request
      print("This script will now attempt to download the MNIST dataset.")
      time.sleep(2)

      os.makedirs('./data/mnist', exist_ok=True)

      for f in ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']:
        urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + f, filename='./data/mnist/' + f)

    mndata = MNIST('./data/mnist')
    mndata.gz = True
    images, labels = mndata.load_training()

    np.random.seed(1)

    batch = list(range(len(images)))
    np.random.shuffle(batch)

    if True:
      recovery_scatterplot([images[i] for i in batch], ms=np.linspace(0, 0.5, num=10))

    # recreate Figure 1 from paper, but with mnist instead of random matrices
    if False:
      resolution = 15
      batch = batch[:1000]
      xs = []
      ys = []
      cs = []
      for img in [images[i] for i in batch]:
        M = np.reshape(img, (28,28))
        r = np.linalg.matrix_rank(M)
        n = M.shape[0]
        min_m = r * (2*n - r)
        for m in np.linspace(min_m, n**2, num=resolution):
          corrupted, omega = corrupt_image(M, 1-(m/n**2))
          d_r = r * (2*n - r)
          xs += [m / n**2]
          ys += [d_r / m]
          print('m/n^2 = {:4.3f} / {:4.3f}'.format(m, n**2))
          print('d_r/m = {:4.3f} / {:4.3f}'.format(d_r, m))

          recovered       = np.round(complete_matrix(corrupted, omega))
          recovery_metric = np.linalg.norm(M - recovered, 'fro') / np.linalg.norm(M, 'fro')
          print('\t', recovery_metric)
          cs += [(recovery_metric, recovery_metric, 0.5)]
      np.savez('mnist_data.npz', xs=xs, ys=ys, cs=cs)
      fig, axs = plt.subplots(1,1, figsize=(5,4))
      axs.set_title('Recovery of matrices from entries')
      axs.scatter(xs, ys, c=cs, s=10)
      axs.set_ylim(0,1)
      axs.set_xlabel('m/n^2')
      axs.set_ylabel('d_r/m')
      plt.savefig('fig_mnist.png')
      plt.show()
        
