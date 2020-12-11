import numpy as np
import random

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os, sys, time, datetime

    if len(sys.argv) != 3:
        print("Need two files.")

    mnist = np.load(sys.argv[1])
    cifar = np.load(sys.argv[2])

    fig, axs = plt.subplots(1,2, figsize=(8,4))
    for name, f, ax in zip(['MNIST', 'CIFAR-10'], [mnist, cifar], axs):
        s = 1 if name == 'CIFAR-10' else 2
        rank = f['rank'][::s]
        xs   = f['xs'][::s]
        dof  = f['dof'][::s]
        cs   = (1 - np.clip(f['cs'], 0, 1)[:,2])[::s]
        new_cs = []
        for c in cs:
            if c < 0.05:
                new_cs.append((0, 1-c, 1-c, 1))
            else:
                new_cs.append((1, 1-c, 1-c, 1))

        ax.set_title('Recovery of {} Images'.format(name))
        ax.scatter(xs, rank, c=new_cs, s=15)
        ax.set_xlabel('Proportion of Corrupted Entries')

        if name == 'MNIST':
            ax.set_ylabel('Image Matrix Rank')
        else:
            ax.set_ylabel('Averaged RGB Channel Rank')
        
    plt.savefig('fig_recov.png')
    plt.show()

