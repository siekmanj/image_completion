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
        X[i,j] = 0

    return X, omega

if __name__ == '__main__':
    import os, sys, datetime, time
    import matplotlib.pyplot as plt

    debug = False
    generate_fig1 = True
    generate_fig2 = False
    generate_fig3 = False

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

    np.random.seed(3)

    batch = list(range(len(images)))
    np.random.shuffle(batch)

    if generate_fig1:
      Ms = [np.reshape(images[i], (28,28)) for i in batch]
      fig = plt.figure(figsize=(8,1))
      axs = []

      rows = 1

      cols=5
      for i in range(rows*cols):
          axs.append(fig.add_subplot(rows, cols, i+1))
          axs[-1].set_xticks([])
          axs[-1].set_yticks([])
          plt.imshow(Ms[i], cmap='gray')

      plt.savefig('examples_mnist.png', transparent=True)
      plt.show()

    if generate_fig2:
      Ms = [np.reshape(images[i], (28,28)) for i in batch]
      fig = plt.figure(figsize=(8,6))
      axs = []

      rows = 3

      cols=3
      for i in range(rows):
          m = [0.25, 0.5, 0.75][i]
          corrupted, omega = corrupt_image(Ms[i], m)
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
          plt.imshow(Ms[i], cmap='gray')

      plt.savefig('comparison.png', transparent=True)
      plt.show()

    if generate_fig3:
      if len(sys.argv) == 2:
        dataname = sys.argv[1]
      else:
        dataname = 'recovery_data.npz'
      if not os.path.isfile(dataname):
        print("creating dataset at '{}'".format(dataname))
        batch = batch[:1000]
        res = 50
        recovery_thresh = 0.01
        ms = np.linspace(0, 1, num=res)

        sampled    = {}
        recoveries = {}
        Ms = [np.reshape(images[i], (28,28)) for i in batch]
        rs = np.linalg.matrix_rank(Ms)
        start = time.time()

        data = {}
        for r in np.unique(rs):
            print("unique rank:", r)
            data[int(r)] = []

        for i, (M, r) in enumerate(zip(Ms, rs)):

          data[int(r)] += [[]]
          for j, m in enumerate(ms):
            corrupted, omega = corrupt_image(M, m)
            recovered        = np.round(complete_matrix(corrupted, omega))
            recovery_metric  = np.clip(np.linalg.norm(M - recovered, 'fro') / np.linalg.norm(M, 'fro'), 0.05, 0.99)

            if recovery_metric < recovery_thresh:
              data[int(r)][-1] += [(1-recovery_metric, 1-recovery_metric, 1-recovery_metric, 1)] # white = recovered, black = not recovered
            else:
              data[int(r)][-1] += [(recovery_metric, 1-recovery_metric, 1-recovery_metric, 1)] # white = recovered, black = not recovered

            completion = (i * len(ms) + j) / (res * len(batch))
            rate = (time.time() - start) / (completion + 1e-3)
            remaining = (1 - completion) * rate
            print('\t{:3d}/{:3d}: {:5.3f}, {} remaining'.format(i * len(ms) + j, res * len(batch), recovery_metric, str(datetime.timedelta(seconds=remaining)).split('.')[0]))

        print("{} elapsed.".format(str(datetime.timedelta(seconds=time.time() - start)).split('.')[0]))
        xs = []
        ys = []
        cs = []
        dof = []
        rank = []
        for r in data.keys():
            data[r] = np.mean(data[r], axis=0)
            for j in range(len(ms)):
                xs   += [ms[j]]
                rank += [r]
                dof  += [4 * (28 - r) * r / (28**2)]
                cs   += [data[r][j]]

        np.savez(dataname, xs=xs, rank=rank, cs=cs, dof=dof)
      else:
        print("loading dataset from '{}'".format(dataname))
        f = np.load(dataname)
        if 'ys' in f:
            rank = f['ys']
            dof  = 4 * (28 - rank) * rank / (28**2)
        elif 'rank' in f and 'dof' in f:
            rank = f['rank']
            dof  = f['dof']

        xs = f['xs']
        cs = f['cs']

      if True: # do DOF, not rank

        fig, axs = plt.subplots(1,1, figsize=(6,4))
        #axs[0].set_title('Recovery of MNIST Images')
        #axs[0].scatter(xs, dof, c=cs, s=20)
        #axs[0].set_xlabel('Proportion of Corrupted Entries')
        #axs[0].set_ylabel('Image Matrix DOF')
        axs.set_title('Recovery of MNIST Images')
        axs.scatter(xs, rank, c=cs, s=20)
        axs.set_xlabel('Proportion of Corrupted Entries')
        axs.set_ylabel('Image Matrix Rank')
        
        print("\\begin{table}[]")
        #print("\\begin{tabular}{l|".join(['l' for _ in np.unique(xs)] + '}'))
        #for 

        print("\\end{tabular}")
        print("\\end{table}")
        plt.savefig('fig_mnist.png')
        plt.show()

