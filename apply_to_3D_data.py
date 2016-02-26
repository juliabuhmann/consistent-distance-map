'''
Regularize distances inferred over EM cube
===============================================================================
This script:
    1) loads the data from its hdf5 file,
    2) builds the corresponding 4D graph model of the problem,
    3) calls the max-flow library to find the optimal graph cut,
    4) plots one slice of the results.
'''



# 0 - Imports and parameters


print "Total python variables memory footprint before everything:", 

### Imports
import numpy as np
import maxflow
from scipy import ndimage as ni
import h5py
import os
from time import time
import sys
import resource


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

MONITOR_MEM = True
# Defines whether memory usage should be printed at various steps.

MAX_DIST =  16
# Maximal distance in the image. Each pixel/voxel is mapped to this many nodes 
# in the final graph. If None, the maximal distance is determined based on the
# image size.


CLIPPING_DIST = 3
# Distance over which cost-function is clipped. cost(x) = cost(3) for all x>3.


KINK = True
WIGGLE = True
# Type of artificial data generated. 
# If WIGGLE is True, a wiggly line is randomly generated from top to bottom.
# If WIGGLE is False and KINK is True, a line with a 90 degrees angle is used.
# Otherwise a straight line is used.


inputfilename = "/Users/abouchar/Downloads/feature_collection_training_cube1.h5"
# hdf5 to load data from


slices = [slice(None), slice(None), slice(21,24)]
# Part of the cube to solve the problem on.
#   slice(None) takes the entire range.
#   slice(start,end) takes range [start,end).









# 1 - Load distance image and ground truth

## 1.a - Load data

    
f = h5py.File(inputfilename, 'r')
# Load data

noisy_distance = f['inference_results'].value[:,:,21:24]
# Inferred distances

ground_truth_distance = f['labels'].value[:,:,21:24]
# Ground truth

shape = list(np.shape(noisy_distance))

f.close()



if MONITOR_MEM:
    print "Memory footprint before graph construction:", get_mem_usg()






# 2 - Generate graph
src_partition = []
offset = 0
count = 0
# Build the graph and optimize. The while loop is there just because the
# initial construction of the graph might end up having a trivial solution
# where all edges from the source are cut and the S partition is empty. This is
# solved by translating the energy by the right amout (offset). There should be
# at most 2 iterations.
while len(src_partition) == 0:
    print "Iteration #" + str(count+1)
    


    print "Adding nodes to graph..."
    MAX_BOUND = np.finfo(np.float).max
    MAX = MAX_BOUND/128
    # Used as weight for pseudo-infinite edges.
    
    t0_graph = time()
    # time graph construction.
        
    max_dist = int(np.ceil(np.sqrt(np.sum(np.power(np.array(shape)-1,2))))) if MAX_DIST == None else MAX_DIST
    # Determine what the maximal distance in the image should be, if not provided.



    ## 2.a - Create Graph with Vertices

    g = maxflow.GraphFloat()
    # Build graph.
    
    
    augmented_shape = shape + [max_dist]
    # Graph shape is image shape + one dimension the size of max_dist.


    nodeids = g.add_grid_nodes(augmented_shape)
    # Add nodes to graph.
    
    


    ## 2.b - Add Edges


    ### Pseudo-infinite edges between non-terminal nodes.
    print "Adding internal edges to graph..."
    
    
    structure = np.zeros((3,3,3,3))
    structure[:,:,:,0] = [[[0, 0, 0],[0, 1, 0],[0, 0, 0]],
                          [[0, 1, 0],[1, 1, 1],[0, 1, 0]],
                          [[0, 0, 0],[0, 1, 0],[0, 0, 0]]]
    # Structure of edges around each node.
    
                          
    weights = np.ones(augmented_shape)*MAX
    # Weights of edges is the same at all nodes (infinite).
    

    g.add_grid_edges(nodeids, weights=weights, structure=structure, symmetric=False)
    # Add internal edges to graph.
    
    
    structure = np.zeros((3,3,3,1))
    structure[:,:,:,0] = [[[0, 0, 0],[0, 1, 0],[0, 0, 0]],
                          [[0, 1, 0],[1, 0, 1],[0, 1, 0]],
                          [[0, 0, 0],[0, 1, 0],[0, 0, 0]]]
    # Structure of edges around nodes at layer 0. Horizontal connections.
    
    
    weights = np.ones(shape+[1])*MAX
    # Infinite weights.
    

    g.add_grid_edges(nodeids[:,:,:,:1], weights=weights, structure=structure, symmetric=False)
    # Add edges to layer 0.



    ### Edges between non-terminal nodes and terminal ones.
    print "Adding terminal edges to graph..."

    #~ weights = np.abs(np.ones(augmented_shape)*range(max_dist) - noisy_distance[:,:,np.newaxis]) # Linear cost.
    weights = np.minimum(np.abs(np.ones(augmented_shape)*range(max_dist) - noisy_distance[:,:,:,np.newaxis]),CLIPPING_DIST) # Linear cost with upper bound.
    # Weights between source/sink and nodes are determined based on the cost for selecting a node.



    weights_diff = np.concatenate((weights[:,:,:,:1], np.diff(weights,axis=3)), axis=3)
    # The actual weights are the total difference between node layers.

    if offset: # In case this is the 2nd iteration, we need to translate our costs.
        weights_diff[0,0,0,0] = weights_diff[0,0,0,0]-offset
        
        
    src_weights = np.abs(np.minimum(weights_diff,0))
    # Source is connected to nodes where weights differences are negative.
    snk_weights = np.maximum(weights_diff,0)
    # Sink is connected to nodes where weights differences are positive.
    
    g.add_grid_tedges(nodeids, src_weights, snk_weights)
    # Add terminal edges to graph.
    
    t1_graph = time()
    print "Done building graph in %.2f seconds." % (t1_graph-t0_graph)
    
    
    # Optionally, monitor python memory usage.
    if MONITOR_MEM:
        print "Memory footprint after graph construction:", get_mem_usg()
    
    
    
    print "Starting optimization..."
    t0_opt = time()
    flow = g.maxflow()
    t1_opt = time()
    "Optimization done in in %.2f seconds." % (t1_opt-t0_opt)

    src_partition = nodeids[np.invert(g.get_grid_segments(nodeids))]
    snk_partition = nodeids[g.get_grid_segments(nodeids)]
    
    offset = np.sum(weights_diff[:,:,:,0])+1
    
    count += 1
    
# 3 - Plot results

print "Plotting..."



### 3.a - Plot one 2D slice

import matplotlib.pyplot as plt

SLICE = 1  # Slice in the solved data subset to plot in 2D.
ROW = 202  # Row in that slice to plot.

fig = plt.figure()
plt.ion()

reg_dist = np.sum(np.invert(g.get_grid_segments(nodeids)),axis=3)-1
# This infers the actual surface height by summing the number of nodes below the cut.
# We remove one, since one node means that the surface passes at level 0.

min_val = 0
max_val = max_dist-1

plt.subplot(1,3,1)
plt.imshow(ground_truth_distance[:,:,1], vmin=min_val, vmax=max_val, interpolation='nearest')
plt.title('Ground Truth')
plt.subplot(1,3,2)
plt.imshow(noisy_distance[:,:,1], vmin=min_val, vmax=max_val, interpolation='nearest')
plt.title('Inferred Distances')
plt.subplot(1,3,3)
im = plt.imshow(reg_dist[:,:,1], vmin=min_val, vmax=max_val, interpolation='nearest')
plt.title('Regularized Distances')

fig.subplots_adjust(right=0.875)
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
fig.colorbar(im,cax=cbar_ax)
plt.show()


### 3.b - Plot a row from that slice.

plt.figure()
plt.hold(True)
plt.plot(ground_truth_distance[ROW,:,SLICE],'g',lw=3)
plt.plot(noisy_distance[ROW,:,SLICE],'r',lw=2)
plt.plot(im[ROW,:,SLICE],'b')
plt.xlim([0,np.shape(im)[1]])
plt.legend(['Ground truth','Noisy distance','Result'])
plt.show()





### 3.c 2D histogram of distances

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.figure()

results = im.flatten()
labels = ground_truth_distance.flatten()

num_of_examples = results.shape[0]
dist = np.mean(np.abs(labels - results))
print dist, 'dist'
print np.median(np.abs(labels - results)), 'median'
plt.hist2d(labels, results, (15, 15), cmap=plt.cm.jet, norm=mpl.colors.LogNorm())
plt.xlabel('ground truth')
plt.ylabel('prediction')
plt.title('2d histogram of predicted distance')
plt.show()

plt.figure()
results = noisy_distance.flatten()
labels = ground_truth_distance.flatten()

num_of_examples = results.shape[0]
dist = np.mean(np.abs(labels - results))
print dist, 'dist'
print np.median(np.abs(labels - results)), 'median'
plt.hist2d(labels, results, (15, 15), cmap=plt.cm.jet, norm=mpl.colors.LogNorm())
plt.xlabel('ground truth')
plt.ylabel('prediction')
plt.title('2d histogram of regularized distance')
plt.show()

pl.ioff()
print "Done"
