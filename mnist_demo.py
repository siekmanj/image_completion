import numpy as np
import random

from complete import complete_matrix, mask_out_matrix
from mnist import MNIST

def mask_out_nonzero(X, proportion):
    """
    Mask out only nonzero entries.
    """
    px_to_remove = proportion * len(X.flatten())

    mask      = np.zeros(X.shape)
    anti_mask = np.zeros(X.shape)

    entries = [(i, j) for i in range(X.shape[0]) for j in range(X.shape[1])]

    np.random.shuffle(entries)

    omega      = nonzero_entries[:int(proportion*len(nonzero_entries))] # entries which stay the same (input to optimization algo)
    anti_omega = nonzero_entries[int(proportion*len(nonzero_entries)):] # entries which are modified

    for (i, j) in omega:
        mask[i, j] = 1

    for (i, j) in anti_omega:
        anti_mask[i, j] = 1

    #return X.copy() * mask + null_matrix, omega
    return mask, anti_mask, omega

if __name__ == '__main__':
    import os, time

    debug = True

    if not os.path.isfile('./data/mnist/train-images-idx3-ubyte.gz'):
        import urllib
        print("This script will now attempt to download the MNIST dataset.")
        time.sleep(2)

        os.makedirs('./data/mnist', exist_ok=True)

        for f in ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']:
            urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + f, filename='./data/mnist/' + f)

    mndata = MNIST('./data/mnist')
    mndata.gz = True
    images, labels = mndata.load_training()

    np.random.seed(0)

    batch = list(range(len(images)))
    np.random.shuffle(batch)
    batch = batch[:100]

    print(batch)

    for i, (img, label) in enumerate(zip([images[i] for i in batch], [labels[i] for i in batch])): 
        M                      = np.reshape(img, (28,28))
        mask, anti_mask, omega = remove_entries(M, 0.1)
        if debug:
          print(np.max(M))
          print("Mask:")
          #print((255*mask).flatten())
          print(MNIST.display(255*(anti_mask).flatten()))
          print(MNIST.display(255*(mask).flatten()))
          print("^^MASK")

        recovered     = complete_matrix(mask*M, omega)

        rank_m         = np.linalg.matrix_rank(M)
        err_frobenius  = np.linalg.norm(M - recovered, 'fro')
        frobenius_diff = np.linalg.norm(M, 'nuc') - np.linalg.norm(recovered, 'nuc')

        print("Original matrix rank: {:2d} | Frobenius Norm of Diff: {:4.3f} | Nuclear Norm Diff {:6.3f}".format(rank_m, err_frobenius, frobenius_diff))

        if debug:
          pass
          #print(MNIST.display(img))
          #print(MNIST.display(masked.flatten()))
          #print(MNIST.display(recovered.flatten()))
