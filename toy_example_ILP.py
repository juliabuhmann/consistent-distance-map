'''
Toy example ILP
===============
This script:
    1) generates a 2D distance image that corresponds to a linear-object and 
       adds noise to it,
    2) builds the corresponding 3D graph model of the problem,
    3) calls the max-flow library to find the optimal graph cut,
    4) plots the results.
'''


# 0 - Imports and parameters

import numpy as np
from scipy import ndimage as ni
from time import time
import pylab as pl
import sys
import os

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


sys.path.append(os.path.join(ROOT_PATH,'src','python'))
sys.path.append(os.path.join('/data/owncloud/MinCutForDistance','pysurfrec','build','python'))
import surface_reconstruction as sr
reload(sr)
MAX_DIST =  None
# Maximal distance in the image. Each pixel/voxel is mapped to this many nodes 
# in the final graph. If None, the maximal distance is determined based on the
# image size.


KINK = True
WIGGLE = False
# Type of artificial data generated. 
# If WIGGLE is True, a wiggly line is randomly generated from top to bottom.
# If WIGGLE is False and KINK is True, a line with a 90 degrees angle is used.
# Otherwise a straight line is used.









# 1 - Generate some noisy distance data

max_dist = MAX_DIST if MAX_DIST != None else 12
column_height = max_dist + 1

## 1.a - Generate data
shape = [150,150]
half_width = int(0.5*(shape[0]-1))
if WIGGLE:
    ground_truth = np.zeros(shape,np.bool)
    ground_truth[0,half_width] = 1
    current = half_width
    for i in range(1,shape[0]):
        step = np.random.randint(-1,2)
        current = np.maximum(0,current+step)
        current = np.minimum(shape[0],current)
        ground_truth[i,current] = 1
        
else:
    # Straight Line
    if not KINK:
        ground_truth = np.zeros(shape,np.bool)
        ground_truth[half_width,:] = 1
    # Kink
    else:
        ground_truth = np.zeros(shape,np.bool)
        ground_truth[half_width,:half_width] = 1
        ground_truth[half_width:,half_width] = 1

ground_truth_distance = ni.morphology.distance_transform_edt(np.invert(ground_truth))


## 1.b - Add noise

np.random.seed(555)
noise_amplitude = 3
#~ noise = np.random.randint(-noise_amplitude,noise_amplitude+1,shape) # Uniformly distributed noise in [-noise_amplitude, noise_amplitude]
noise = np.random.randn(shape[0],shape[1])*noise_amplitude # Gaussian noise ~ N(0,noise_amplitude)

noisy_distance = np.minimum(np.maximum(ground_truth_distance+noise,0),max_dist)

#print " ".join([str(n) for n in noisy_distance])
## 2.a - Set weights
augmented_shape = shape + [column_height]
slice_ = [slice(None) for _ in range(len(noisy_distance.shape))] + [np.newaxis]
print augmented_shape, column_height, slice_
weights = np.abs(np.ones(augmented_shape)*range(column_height) - noisy_distance[slice_])
#print weights
value = sr.solve_via_ILP(weights, max_gradient=1)
wf = weights.reshape(np.prod(shape),column_height)
print sum([w[v] for v,w in zip(value.flatten(),wf)])
print "solving second"
value2 = sr.reconstruct_surface(noisy_distance,'/data/dump/tmp0.txt','/data/dump/tmp1.txt',os.path.join(ROOT_PATH,'src','cpp','graph_cut_Linux'),max_dist=max_dist,cost_fun='linear',overwrite=True,verbose=True)
print sum([w[v] for v,w in zip(value2.flatten(),wf)])
pl.subplot(2,3,1)
pl.imshow(ground_truth_distance,vmin=0,vmax=max_dist,interpolation='nearest')
pl.title('Ground Truth')
pl.subplot(2,3,2)
pl.imshow(noisy_distance,vmin=0,vmax=max_dist,interpolation='nearest')
pl.title('Noisy Distance')
pl.subplot(2,3,4)
pl.imshow(value,vmin=0,vmax=max_dist,interpolation='nearest')
pl.title('ILP estimation')
pl.subplot(2,3,5)
pl.imshow(value2,vmin=0,vmax=max_dist,interpolation='nearest')
pl.title('MinCut estimation')
pl.subplot(2,3,6)
pl.imshow(value-value2,interpolation='nearest')
pl.title('ILP estimation - MinCut estimation')
print "Max gradient ILP:", np.maximum(np.max(np.diff(value,axis=0)),np.max(np.diff(value,axis=1))), ", Graph-Cut:", np.maximum(np.max(np.diff(value2,axis=0)),np.max(np.diff(value2,axis=1)))
pl.show()

