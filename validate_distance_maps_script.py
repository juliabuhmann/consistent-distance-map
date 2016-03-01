'''
Validates EM distance map
===============================================================================
This script:
    1) reads in h5 files
    2) calculates some error metric
    3) plots 2d histogram
'''

import h5py
import validate_distance_maps as vdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

inputfilename = "/raid/julia/projects/LSTM/membrane_distance/20160208/" \
               "inference_results/202452/feature_collection_testing.h5"
PLOTTING = True


f = h5py.File(inputfilename, 'r')
gt_dm = f['labels'].value
f.close()

f = h5py.File(inputfilename, 'r')
noisy_dm = f['inference_results'].value
f.close()

L1 = vdm.calculate_L1(gt_dm, noisy_dm)
L2 = vdm.calculate_L2(gt_dm, noisy_dm)

perc = vdm.calculate_perc_of_correct(gt_dm.astype(np.int), noisy_dm.astype(np.int))
print 'error noisy versus ground truth'
print 'L1: %0.4f    L2: %0.4f     perc_correct: %0.4f' %(L1, L2, perc)
if PLOTTING:
    results = noisy_dm.flatten()
    labels = gt_dm.flatten()
    plt.hist2d(labels, results, (15, 15), cmap=plt.cm.jet, norm=mpl.colors.LogNorm())
    plt.xlabel('ground truth')
    plt.ylabel('noisy distance map')
    plt.title('2d histogram of noisy distance')
    plt.show()