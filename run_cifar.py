# Generates the CIFAR-10 portion of Fig. 4

import numpy as np
import random
import os

from cifar10_web import cifar10
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

def get_cifar(download=False):
    if not os.path.isfile('./data/cifar/cifar-10-binary.tar.gz'):
        if download:
            print("\n\n")
            print("This script will now attempt to download the cifar10 dataset.")
            print("Depending on your internet speed, this may take several minutes.")
            images, _, _, _ = cifar10(path='data/cifar')
        else:
            print("\n\n")
            print("This script is going to load a small, pre-cached version of the cifar10 dataset.")
            print("To disable this and download the full dataset, call get_cifar(download=True)")
            images = np.load('data/cifar/small_cifar.npy')
    else:
        images, _, _, _ = cifar10(path='data/cifar')

    return images

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os, sys, time, datetime

    images = get_cifar()
    batch = list(range(len(images)))
    np.random.shuffle(batch)

    if len(sys.argv) == 2:
        dataname = sys.argv[1]
    else:
        dataname = 'cifar_recovery.npz'

    if not os.path.isfile(dataname):
        print("creating dataset at '{}'".format(dataname))
        if False: # takes 10-20 hours
            batch = batch[:1000]
            res = 25
        else: # takes a few minutes
            batch = batch[:10]
            res = 4
        recovery_thresh = 0.01
        ms = np.linspace(0, 1, num=res)

        sampled    = {}
        recoveries = {}
        start = time.time()

        data = {}
        for idx, i in enumerate(batch): 
            img = images[i]
            # extract the r, g, b channels from the dataset
            r = [img[j] for j in range(1024)]
            g = [img[j] for j in range(1024, 2048)]
            b = [img[j] for j in range(2048, 3072)]

            # reshape from flat vector into 32x32
            r = np.reshape(r, (32,32))
            g = np.reshape(g, (32,32))
            b = np.reshape(b, (32,32))

            # calculate mean rank of three color channels
            rank = np.mean([np.linalg.matrix_rank(ch) for ch in (r, g, b)])

            if rank not in data:
                data[rank] = [[]]
            else:
                data[rank] += [[]]

            for j, m in enumerate(ms):
                # perform matrix completion on the corrupted channels, one at a time.
                r_bad, g_bad, b_bad, omega = corrupt_channels(r,g,b,m)
                r_hat = complete_matrix(r_bad, omega)
                g_hat = complete_matrix(g_bad, omega)
                b_hat = complete_matrix(b_bad, omega)

                recovery_metric = 0
                for M, R in [(r, r_hat), (g, g_hat), (b, b_hat)]:
                    recovery_metric += (np.linalg.norm(M - R, 'fro') / np.linalg.norm(M, 'fro')) / 3 # average recover metric over the 3 channels

                data[rank][-1] += [recovery_metric]

                completion = (idx * len(ms) + j) / (res * len(batch))
                rate = (time.time() - start) / (completion + 1e-3)
                remaining = (1 - completion) * rate
                print('\t{:3d}/{:3d}: {:5.3f}, {} remaining'.format(idx * len(ms) + j, res * len(batch), recovery_metric, str(datetime.timedelta(seconds=remaining)).split('.')[0]))

        print("{} elapsed.".format(str(datetime.timedelta(seconds=time.time() - start)).split('.')[0]))
        xs = []
        ys = []
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
    xs = f['xs']
    cs = np.clip(f['cs'], 0.01, 1)

    cs = np.clip(cs, 0, 1)
    fig, axs = plt.subplots(1,1, figsize=(6,4))
    axs.set_title('Recovery of CIFAR-10 Images')
    new_cs = []
    for c in cs:
        if c < 0.10:
            new_cs.append((0, 1-c, 1-c, 1))
        else:
            new_cs.append((1, 1-c, 1-c, 1))
    axs.scatter(xs, rank, c=new_cs, s=20)
    axs.set_xlabel('Proportion of Corrupted Entries')
    axs.set_ylabel('Averaged RGB Channel Rank')

    plt.savefig('figs/fig_cifar.png')
    plt.show()
