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
import results_evaluation as evaluate
reload(sr)
MAX_DIST =  12
# Maximal distance in the image. Each pixel/voxel is mapped to this many nodes 
# in the final graph. If None, the maximal distance is determined based on the
# image size.



KINK = True
WIGGLE = False
# Type of artificial data generated. 
# If WIGGLE is True, a wiggly line is randomly generated from top to bottom.
# If WIGGLE is False and KINK is True, a line with a 90 degrees angle is used.
# Otherwise a straight line is used.

shape = [25,25]


def artificial_data(shape, noise_amplitude=3, max_dist=None, line_type='wiggle',seed=555):
    

    # 1 - Generate some noisy distance data

    max_dist = max_dist if max_dist != None else 12
    column_height = max_dist + 1

    ## 1.a - Generate data
    if isinstance(shape,int):
        shape = [shape,shape]
    elif isinstance(shape,list):
        pass
    else:
        raise Exception()
        
    half_width = int(0.5*(shape[0]-1))

    if line_type.upper() == 'WIGGLE':
        
        ground_truth = np.zeros(shape,np.bool)
        ground_truth[0,half_width] = 1
        current = half_width
        for i in range(1,shape[0]):
            step = np.random.randint(-1,2)
            current = np.maximum(0,current+step)
            current = np.minimum(shape[0],current)
            ground_truth[i,current] = 1
            
    elif line_type.upper() == 'STRAIGHT':
            ground_truth = np.zeros(shape,np.bool)
            ground_truth[half_width,:] = 1
        # Kink
    else:
            ground_truth = np.zeros(shape,np.bool)
            ground_truth[half_width,:half_width] = 1
            ground_truth[half_width:,half_width] = 1

    ground_truth_distance = ni.morphology.distance_transform_edt(np.invert(ground_truth))


    ## 1.b - Add noise

    np.random.seed(seed)
    #~ noise = np.random.randint(-noise_amplitude,noise_amplitude+1,shape) # Uniformly distributed noise in [-noise_amplitude, noise_amplitude]
    noise = np.random.randn(shape[0],shape[1])*noise_amplitude # Gaussian noise ~ N(0,noise_amplitude)

    noisy_distance = np.minimum(np.maximum(ground_truth_distance+noise,0),max_dist)

    return noisy_distance




times = []

for size in [5,9,11,13,15,17,19,21,23,25,27,29,31]:
    print "Timing optimization for image of size " + str(size) + 'x' + str(size) + '.'
    times_ = []
    for rep in range(3):
        
        noisy_distance = artificial_data(size, noise_amplitude=3, max_dist=MAX_DIST, line_type='wiggle', seed=rep)
        column_height = MAX_DIST+1

        augmented_shape = [size, size, column_height]
        slice_ = [slice(None) for _ in range(len(noisy_distance.shape))] + [np.newaxis]
        
        weights = np.abs(np.ones(augmented_shape)*range(column_height) - noisy_distance[slice_])

        print "Starting optimization for repetition", rep+1
        t0 = time()
        value_m = sr.solve_via_ILP(weights, max_gradient=1, enforce_minimum=True)
        times_.append(time()-t0)
        print "Done in " + str(times_[-1]) + " seconds."
    times.append(times_)


pl.plot([s*s for s in sizes, np.mean(times,axis=1), '.'])
pl.show()



value_m = sr.solve_via_ILP(weights, max_gradient=1, enforce_minimum=True)




#print " ".join([str(n) for n in noisy_distance])
## 2.a - Set weights
augmented_shape = shape + [column_height]
slice_ = [slice(None) for _ in range(len(noisy_distance.shape))] + [np.newaxis]
print augmented_shape, column_height, slice_
weights = np.abs(np.ones(augmented_shape)*range(column_height) - noisy_distance[slice_])






value_m = sr.solve_via_ILP(weights, max_gradient=1, enforce_minimum=True)
value = sr.solve_via_ILP(weights, max_gradient=1, enforce_minimum=False)
wf = weights.reshape(np.prod(shape),column_height)

print "solving second"
#~ value2 = sr.reconstruct_surface(noisy_distance,'/data/dump/tmp0.txt','/data/dump/tmp1.txt',os.path.join(ROOT_PATH,'src','cpp','graph_cut'),max_dist=max_dist,cost_fun='linear',overwrite=True,verbose=True)



print "Total Cost basic ILP:", sum([w[v] for v,w in zip(value.flatten(),wf)])
print "Total Cost ILP without non-0 minima:", sum([w[v] for v,w in zip(value_m.flatten(),wf)])
#print "Total Cost basic Min-Cut:", sum([w[v] for v,w in zip(value2.flatten(),wf)])


evaluate.compare_scores(ground_truth_distance, [value, value_m], ['ILP','ILP with minima at 0 only'])


pl.subplot(2,3,1)
pl.imshow(ground_truth_distance,vmin=0,vmax=max_dist,interpolation='nearest')
pl.title('Ground Truth')
pl.subplot(2,3,2)
pl.imshow(noisy_distance,vmin=0,vmax=max_dist,interpolation='nearest')
pl.title('Noisy Distance')
pl.subplot(2,3,3)
pl.imshow(value_m,vmin=0,vmax=max_dist,interpolation='nearest')
pl.title('ILP minima at 0 only')
pl.subplot(2,3,4)
pl.imshow(value,vmin=0,vmax=max_dist,interpolation='nearest')
pl.title('ILP estimation')
#~ pl.subplot(2,3,5)
#~ pl.imshow(value2,vmin=0,vmax=max_dist,interpolation='nearest')
#~ pl.title('MinCut estimation')
pl.subplot(2,3,6)
pl.imshow(value-value_m,interpolation='nearest')
pl.title('ILP estimations difference')
#~ print "Max gradient ILP:", np.maximum(np.max(np.diff(value,axis=0)),np.max(np.diff(value,axis=1))), ", Graph-Cut:", np.maximum(np.max(np.diff(value2,axis=0)),np.max(np.diff(value2,axis=1)))
pl.show()







#print weights
#~ value_m = sr.solve_via_ILP(weights, max_gradient=1, enforce_minimum=True)
#~ value = sr.solve_via_ILP(weights, max_gradient=1)
#~ wf = weights.reshape(np.prod(shape),column_height)
#~ print sum([w[v] for v,w in zip(value.flatten(),wf)])
#~ print "solving second"
#~ value2 = sr.reconstruct_surface(noisy_distance,'/data/dump/tmp0.txt','/data/dump/tmp1.txt',os.path.join(ROOT_PATH,'src','cpp','graph_cut_Linux'),max_dist=max_dist,cost_fun='linear',overwrite=True,verbose=True)
#~ print sum([w[v] for v,w in zip(value2.flatten(),wf)])
#~ pl.subplot(2,3,1)
#~ pl.imshow(ground_truth_distance,vmin=0,vmax=max_dist,interpolation='nearest')
#~ pl.title('Ground Truth')
#~ pl.subplot(2,3,2)
#~ pl.imshow(noisy_distance,vmin=0,vmax=max_dist,interpolation='nearest')
#~ pl.title('Noisy Distance')
#~ pl.subplot(2,3,3)
#~ pl.imshow(value_m,vmin=0,vmax=max_dist,interpolation='nearest')
#~ pl.title('ILP estimation')
#~ pl.subplot(2,3,5)
#~ pl.imshow(value,vmin=0,vmax=max_dist,interpolation='nearest')
#~ pl.title('ILP estimation')
#~ pl.subplot(2,3,5)
#~ pl.imshow(value2,vmin=0,vmax=max_dist,interpolation='nearest')
#~ pl.title('MinCut estimation')
#~ pl.subplot(2,3,6)
#~ pl.imshow(value-value_m,interpolation='nearest')
#~ pl.title('ILP estimation - ILP estimation')
#~ print "Max gradient ILP:", np.maximum(np.max(np.diff(value,axis=0)),np.max(np.diff(value,axis=1))), ", Graph-Cut:", np.maximum(np.max(np.diff(value2,axis=0)),np.max(np.diff(value2,axis=1)))
#~ pl.show()

