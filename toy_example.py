'''
Toy example
===========
This script:
    1) generates a 2D distance image that corresponds to a linear-object and 
       adds noise to it,
    2) builds the corresponding 3D graph model of the problem,
    3) calls the max-flow library to find the optimal graph cut,
    4) plots the results.
'''


# 0 - Imports and parameters

import numpy as np
import maxflow
from scipy import ndimage as ni
from time import time

MAX_DIST =  None
# Maximal distance in the image. Each pixel/voxel is mapped to this many nodes 
# in the final graph. If None, the maximal distance is determined based on the
# image size.


KINK = True
WIGGLE = True
# Type of artificial data generated. 
# If WIGGLE is True, a wiggly line is randomly generated from top to bottom.
# If WIGGLE is False and KINK is True, a line with a 90 degrees angle is used.
# Otherwise a straight line is used.









# 1 - Generate some noisy distance data

## 1.a - Generate data
shape = [15,15]
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
noise_amplitude = 1
noise = np.random.randint(-noise_amplitude,noise_amplitude+1,shape)
noise = np.random.randn(shape[0],shape[1])*noise_amplitude

noisy_distance = np.maximum(ground_truth_distance+noise,0)










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
    
    
    MAX_BOUND = np.finfo(np.float).max
    MAX = MAX_BOUND/128 # 
    # Used as weight for pseudo-infinite edges.    
    
    
    t0_graph = time()
    # time graph construction.
    
    # 2 - Generate graph

    ## 2.a - Create Graph and add Vertices
    
    max_dist = int(np.ceil(np.sqrt(np.sum(np.power(np.array(shape)-1,2))))) if MAX_DIST == None else MAX_DIST
    # Determine what the maximal distance in the image should be, if not provided.


    g = maxflow.GraphFloat()
    # Build graph.
    

    augmented_shape = shape + [max_dist]
    # Graph shape is image shape + one dimension the size of max_dist.

    nodeids = g.add_grid_nodes(augmented_shape)
    # Add nodes to graph.
    
    
    
    

    ## 2.b - Add Edges

    ### Pseudo-infinite edges between non-terminal nodes.
    
    structure = np.zeros((3,3,3))
    structure[:,:,2] = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
    structure[:,:,1] = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
    structure[:,:,0] = [[0, 1, 0],[1, 1, 1],[0, 1, 0]]
    # Structure of edges around each node.
    
    weights = np.ones(augmented_shape)*MAX
    # Weights of edges is the same at all nodes (infinite).

    g.add_grid_edges(nodeids, weights=weights, structure=structure, symmetric=False)
    # Add internal edges to graph.



    ### Bottom layer of nodes should be linked horizontally.
    
    structure = np.zeros((3,3,1))
    structure[:,:,0] = [[0, 1, 0],[1, 0, 1],[0, 1, 0]]
    # Horizontal structure.
    
    weights = np.ones(shape+[1])*MAX
    # Infinite weights.
    
    g.add_grid_edges(nodeids[:,:,:1], weights=weights, structure=structure, symmetric=False)
    # Add edges to bottom layer.



    ### Edges between non-terminal nodes and terminal ones.

    weights = np.abs(np.ones(augmented_shape)*range(max_dist) - noisy_distance[:,:,np.newaxis]) # Linear cost.
    #~ weights = np.minimum(np.abs(np.ones(augmented_shape)*range(max_dist) - noisy_distance[:,:,:,np.newaxis]),DIST_CAP) # Linear cost with upper bound.
    # Weights between source/sink and nodes are determined based on the cost for selecting a node.


    weights_diff = np.concatenate((weights[:,:,:1], np.diff(weights,axis=2)), axis=2)
    # The actual weights are the total difference between node layers.

    if offset: # In case this is the 2nd iteration, we need to translate our costs.
        weights_diff[0,0,0] = weights_diff[0,0,0]-offset
        
        
    src_weights = np.abs(np.minimum(weights_diff,0))
    # Source is connected to nodes where weights differences are negative.
    snk_weights = np.maximum(weights_diff,0)
    # Sink is connected to nodes where weights differences are positive.
    
    g.add_grid_tedges(nodeids, src_weights, snk_weights)
    # Add edges to graph.
    
    t1_graph = time()
    print "Done building graph in %.2f seconds." % (t1_graph-t0_graph)
    
    ## 2.c Perform optimization
    print "Starting optimization..."
    t0_opt = time()
    flow = g.maxflow()
    t1_opt = time()
    print "Optimization done in in %.2f seconds." % (t1_opt-t0_opt)

    src_partition = nodeids[np.invert(g.get_grid_segments(nodeids))]
    snk_partition = nodeids[g.get_grid_segments(nodeids)]
    
    offset = np.sum(weights_diff[:,:,0])+1
    
    count += 1
    
    
    
# 3 - Plot results

print "Plotting..."
PLOT_3D = False
# Mostly for debugging purposes. Creates a 3D scatter plot of all nodes in the
# graph colored by the partition they belong to.
if PLOT_3D:
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt


    pos = np.array([np.where(nodeids==elmt) for elmt in src_partition])

    pos_n = np.array([np.where(nodeids==elmt) for elmt in snk_partition])

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.hold(True)

    ax.scatter(pos_n[:,0,:].flatten(),pos_n[:,1,:].flatten(),pos_n[:,2,:].flatten(),c='g')

    ax.scatter(pos[:,0,:].flatten(),pos[:,1,:].flatten(),pos[:,2,:].flatten(),c='r')

    plt.show()
    
    
import matplotlib.pyplot as plt

plt.figure()
plt.ion()
im = np.sum(np.invert(g.get_grid_segments(nodeids)),axis=2)-1
#pl.subplot(1,2,1)
#pl.imshow(noisy_distance,interpolation='nearest')
#pl.subplot(1,2,2)
H = np.shape(im)[0]
spacer = np.zeros((H,3))
IM = np.concatenate((ground_truth_distance, spacer, noisy_distance, spacer, im), axis=1)
plt.imshow(IM,interpolation='nearest')
plt.title('True Distance   -    Distance with Noise   -   Result')
#    pl.colorbar()
plt.show()

plt.figure()
plt.hold(True)
plt.plot(ground_truth_distance[half_width,:],'g')
plt.plot(noisy_distance[half_width,:],'r')
plt.plot(im[half_width,:],'b')
plt.legend(['Ground truth','Noisy distance','Result'])
plt.show()
plt.ioff()
print "Done"
