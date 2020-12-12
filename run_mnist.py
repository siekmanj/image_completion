import numpy as np
import random

import os
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

def get_mnist():
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
    return [np.reshape(i, (28,28)) for i in images]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys, datetime, time

    np.random.seed(3)

    images = get_mnist()
    batch = list(range(len(images)))
    np.random.shuffle(batch)

    if len(sys.argv) == 2:
        dataname = sys.argv[1]
    else:
        dataname = 'recovery_data.npz'

    if not os.path.isfile(dataname):
        print("creating dataset at '{}'".format(dataname))
        if False: # full scale test, takes quite a long time
            batch = batch[:1000]
            res = 25
        else: # quick demo, takes 1-2 minutes
            batch = batch[:10]
            res = 5
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
              recovery_metric  = np.clip(np.linalg.norm(M - recovered, 'fro') / np.linalg.norm(M, 'fro'), 0.01, 0.99)

              data[int(r)][-1] += [recovery_metric]

              completion = (i * len(ms) + j) / (res * len(batch))
              rate = (time.time() - start) / (completion + 1e-3)
              remaining = (1 - completion) * rate
              print('\t{:3d}/{:3d}: {:5.3f}, {} remaining'.format(i * len(ms) + j, res * len(batch), recovery_metric, str(datetime.timedelta(seconds=remaining)).split('.')[0]))

      print("{} elapsed.".format(str(datetime.timedelta(seconds=time.time() - start)).split('.')[0]))
      xs = []
      cs = []
      rank = []
      for r in data.keys():
          data[r] = np.mean(data[r], axis=0)
          for j in range(len(ms)):
              xs   += [ms[j]]
              rank += [r]
              cs   += [data[r][j]]
      np.savez(dataname, xs=xs, rank=rank, cs=cs)

    print("loading dataset from '{}'".format(dataname))
    f = np.load(dataname)
    rank = f['rank']
    xs   = f['xs']
    cs   = f['cs']

    fig, axs = plt.subplots(1,1, figsize=(6,4))
    axs.set_title('Recovery of MNIST Images')
    new_cs = []
    for c in cs:
        if c < 0.10:
            new_cs.append((0, 1-c, 1-c, 1))
        else:
            new_cs.append((1, 1-c, 1-c, 1))

    axs.scatter(xs, rank, c=new_cs, s=20)
    axs.set_xlabel('Proportion of Corrupted Entries')
    axs.set_ylabel('Image Matrix Rank')
    plt.legend()
    plt.savefig('fig_mnist.png')
    plt.show()
