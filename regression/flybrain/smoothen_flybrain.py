'''
Regularize distances inferred over EM cube
===============================================================================
This script:
    1) loads the data from its tiff file,
    2) builds the corresponding 4D graph model of the problem,
    3) calls the max-flow library to find the optimal graph cut,
    4) plots one slice of the results.
'''



# 0 - Imports and parameters


print "Total python variables memory footprint before everything:", 

### Imports
import numpy as np
from tifffile import imread, imsave
import os
from time import time
import sys
import resource

ROOT_PATH = os.path.join( os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir )

sys.path.append(os.path.join(ROOT_PATH,'src','pyhton'))

import surface_reconstruction as sr
import results_evaluation as evaluate
reload(evaluate)
### Function definitions
def sizeof_fmt(num, suffix='B'):
    # Format size as human readable.
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Y', suffix)
    
def get_mem_usg():
    '''Returns a formatted string that indicates how much memory is currently
    used by python.
    '''
    return sizeof_fmt(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)



### Constants
WRITE_SMOOTHED = True

MONITOR_MEM = True
# Defines whether memory usage should be printed at various steps.

MAX_DIST =  16
# Maximal distance in the image. Each pixel/voxel is mapped to this many nodes 
# in the final graph. If None, the maximal distance is determined based on the
# image size.


CLIPPING_DIST = 3
# Distance over which cost-function is clipped. cost(x) = cost(3) for all x>3.


ROOT_DATA = '/Users/abouchar/ownCloud/ProjectRegSeg/data/Flybrain'


path_output = os.path.join(ROOT_DATA,'Smoothed_Predictions')

path_predictions = os.path.join(ROOT_DATA,'Predictions')
path_GT = os.path.join(ROOT_DATA,'GroundTruth')

fn_predictions = [f for f in os.listdir(path_predictions) if f.endswith('.tif')]
fn_GT = [f for f in os.listdir(path_GT) if f.endswith('.tif')] 
        
#~ cost_fun = 'lin_clipped'
cost_fun = 'weighted_asymmetric'

input_filename = os.path.join(path_predictions,fn_predictions[1])
output_filename = os.path.join(path_output,fn_predictions[1][:-8] + 'smt_pred_' + cost_fun + '.tif')
GT_filename = os.path.join(path_GT,fn_GT[1])


# Part of the cube to solve the problem on.
#   slice(None) takes the entire range.
#   slice(start,end) takes range [start,end).


tmp1 = '/Users/abouchar/Desktop/dump/tmp.txt'
tmp2 = '/Users/abouchar/Desktop/dump/tmp_out.txt'

prog = os.path.join(ROOT_PATH,'src','cpp','graph_cut')




# 1 - Load distance image and ground truth

## 1.a - Load data

max_dist = 15


slices = [slice(None), slice(None), slice(None)]


# Load data
    
noisy_distance = np.minimum(np.maximum(np.round( imread(input_filename) ).astype(np.int),0),max_dist)[slices]
# Inferred distances

ground_truth_distance = np.minimum(np.maximum(np.round(imread(GT_filename)).astype(np.int),0),max_dist)[slices]
# Ground truth

shape = list(np.shape(noisy_distance))



if MONITOR_MEM:
    print "Memory footprint before graph construction:", get_mem_usg()





# 2 - Solve Graph-Cut
if os.path.isfile(output_filename):
    results = imread(output_filename)
else:
    results = sr.reconstruct_surface_VCE(noisy_distance, tmp1, tmp2, prog, max_dist=max_dist, cost_fun=cost_fun, verbose=True, overwrite=True)


    
# 3 - Plot results

print "Plotting..."



### 3.a - Plot one 2D slice

import pylab as pl
cols = 3
rows = 3
sl = np.linspace(0,int(shape[-1]-1),5)[1:-1].astype(int)

evaluate.compare_scores(ground_truth_distance, [noisy_distance, results], ['Prediction','Smoothed Prediction'])

print np.sum(ground_truth_distance==0), np.sum(noisy_distance==0), np.sum(results==0),

#evaluate.compare_scores(ground_truth_distance<1, [noisy_distance<1, results<1], ['Prediction','Smoothed Prediction'])

if WRITE_SMOOTHED and not os.path.isfile(output_filename):
    imsave(output_filename,results.astype(np.int32),compress=1)

pl.figure()
for R in range(rows):
    pl.subplot(rows,cols,R*cols+1)
    pl.imshow(ground_truth_distance[:,:,sl[R]],vmin=0,vmax=max_dist)
    pl.xticks([])
    pl.yticks([])
    if R == 0:
        pl.ylabel('Slice #' + str(sl[R]))
    pl.title('Ground Truth')
    pl.subplot(rows,cols,R*cols+2)
    pl.imshow(noisy_distance[:,:,sl[R]],vmin=0,vmax=max_dist)
    pl.xticks([])
    pl.yticks([])
    pl.title('Prediction')
    pl.subplot(rows,cols,R*cols+3)
    pl.imshow(results[:,:,sl[R]],vmin=0,vmax=max_dist)
    pl.xticks([])
    pl.yticks([])
    pl.title('Smoothed Prediction')
pl.show()
