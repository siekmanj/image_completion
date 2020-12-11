# Recreates the original Figure 1 from the referenced paper.
import numpy as np
import random

from complete import complete_psd_symmetric

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os, sys, time, datetime

    pix_res = 32,32
    trials = 20

    img = np.zeros(pix_res)
    start = time.time()
    for n in [28]:
        for x, dr_over_m in enumerate(np.linspace(0, 1, num=pix_res[0])):
            for y, m_over_nsq in enumerate(np.linspace(0, 1, num=pix_res[1])):
                for trial in range(trials):
                    m = int((m_over_nsq * n**2))
                    r = int(np.max([(-(2*n+1) + np.sqrt(max(0, (2 * n + 1)**2 - 4 * (-1) * (-2 * dr_over_m*m))))/(-2),
                                    (-(2*n+1) - np.sqrt(max(0, (2 * n + 1)**2 - 4 * (-1) * (-2 * dr_over_m*m))))/(-2)]))
                    M_f = np.random.randn(n, r)

                    M = M_f @ M_f.T
                    omega = random.sample([(i, j) for i in range(M.shape[0]) for j in range(M.shape[1])], m)

                    X = complete_psd_symmetric(M.copy(), omega)

                    if X is not None:
                        metric = np.clip(np.linalg.norm(X - M, 'nuc') / np.linalg.norm(M, 'nuc'), 0, 1)
                    else:
                        metric = 1

                    if metric < 1e-3:
                        img[x,y] += 1

                    completion = (x * pix_res[1] * trials + y * trials + trial) / (pix_res[0] * pix_res[1] * trials)
                    rate = (time.time() - start) / (completion + 1e-3)
                    remaining = (1 - completion) * rate
                    print('\t{:3d}/{:3d}: m={:3.2f}, metric={:5.3f}, {} remaining'.format(x * pix_res[1] * trials + y * trials + trial, trials * pix_res[0] * pix_res[1], m_over_nsq, metric, str(datetime.timedelta(seconds=remaining)).split('.')[0]))
        img = img/trials
        plt.imshow(img, cmap='gray', extent=[0,1,0,1])
        plt.xlabel('m/n^2')
        plt.ylabel('d/m')
        plt.savefig('recreation_fig_2.png')
        plt.show()
