# Recreates the original Figure 1 from the referenced paper.
import numpy as np
import random

from complete import complete_psd_symmetric

if __name__ == '__main__':
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'sans-serif'


    import matplotlib.pyplot as plt
    import os, sys, time, datetime

    pix_res = 16,16
    trials = 5

    img = np.zeros(pix_res)
    start = time.time()
    if not os.path.isfile('backup.npy'):
        for n in [16]:
            for x, m_over_Dn in enumerate(np.linspace(0.05, 1, num=pix_res[0])):
                for y, dr_over_m in enumerate(np.linspace(0.11, 1, num=pix_res[1])):
                    for trial in range(trials):
                        m = int((m_over_Dn) * 0.5 * (n**2 + n))
                        r = max(1, int(((2*n+1) - np.sqrt((2*n+1)**2 - 8 * dr_over_m*m))/2))
                        M_f = np.random.randn(n, r)

                        M = M_f @ M_f.T
                        omega = random.sample([(i, j) for i in range(M.shape[0]) for j in range(M.shape[1])], m)

                        X = None
                        while X is None:
                            X = complete_psd_symmetric(M.copy(), omega)
                            if X is None:
                                print("Repeating opt, got none")

                        if X is not None:
                            metric = np.clip(np.linalg.norm(X - M, 'nuc') / (np.linalg.norm(M, 'nuc') + 1e-2), 0, 1)
                        else:
                            metric = 1

                        if metric < 1e-3:
                            img[y,x] += 1

                        completion = (x * pix_res[1] * trials + y * trials + trial) / (pix_res[0] * pix_res[1] * trials)
                        rate = (time.time() - start) / (completion + 1e-3)
                        remaining = (1 - completion) * rate

                        print('\t{:3d}/{:3d}: rank {:2d}, m={:4d}/{:4d}, img[{:3.2f},{:3.2f}]=metric={:5.3f}, {} remaining'.format(x * pix_res[0] * trials + y * trials + trial, trials * pix_res[1] * pix_res[1], r, m, int(n*(n+1)/2), m_over_Dn, dr_over_m, metric, str(datetime.timedelta(seconds=remaining)).split('.')[0]))
            img = img/trials
            np.save('backup.npy', img)
    img = np.load('backup.npy')
    plt.imshow(img, cmap='gray', origin='lower', extent=[0,1,0.1,1])
    plt.title('Reproduction of Fig. 2 from Candes et. al.', fontsize=20)
    plt.xlabel(r'$m/D_n$', fontsize=18)
    plt.ylabel(r'$d_r/m$', fontsize=18)
    plt.savefig('figs/repro_fig2.png')
    plt.show()
