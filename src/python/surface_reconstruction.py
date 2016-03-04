### Imports
import numpy as np
#import pymaxflow
from scipy import ndimage as ni
import h5py
import os
from time import time
import sys
import resource
from subprocess import call, STDOUT
import validate_distance_maps as vdm


def prepare_problem(distance_map, path_output, max_dist=None, sampling=None, overwrite=False, cost_fun='lin_clipped',clipping_dist=4):
    '''
    # Parameters
    ============
    
    distance_map: numpy.ndarray
        Array containing the regressed distance.
        
    path_output: string
        Path to the file where to write the graph data that will be read by the
        C++ program.
        
    max_dist: int
        Maximum distance that can be predicted. If not provided, the maximal
        value in 'distance_map' is used.
        
    sampling: list of ints
        Sampling period in the image. The periods should be integers defined
        relatively to each other (e.g. [1,1,2] corresponds to 3D data where
        voxels are twice as large along Z). If not provided, assumes isotropic
        data.
        
    overwrite: bool
        Defines whether an existing output file can be overwritten with the new
        output. Default is False.
        
    cost_fun: string
        One of
        
        
    # Example
    =========
    
    
    import h5py
    
    f = h5py.File(inputfilename, 'r')
    
    predicted_distance = f['inference_results'].value
    
    sampling = [1,1,2] # Z has half resolution.
        
    f.close()
    
    output_path = "/my/choice/of/path/graph_data.txt"
    
    prepare_problem(predicted_distance, output_path, max_dist=15, sampling=sampling)
    
        
    '''
    shape = list(np.shape(distance_map))
    
    if sampling != None:
        assert len(sampling) == len(shape)
        assert min(sampling) >= 1

    if max_dist == None:
        max_dist = int(np.max(distance_map))
        print "Max distance not defined. Using " + str(max_dist) + "." 
    
    if cost_fun == None:
        cost_fun='lin_clipped'
        
    column_height = max_dist+1 # ranges from 0 to max_dist

    if os.path.exists(path_output) and not overwrite:
        raise Exception(path_output + ' already exists. Switch "overwrite" argument to True to replace it.')


    with open(path_output,'wb') as f_out:
        
        if sampling == None: # If not defined, assume isotropic sampling.
            sampling = [1 for _ in range(len(shape)+1)]
            
        
        # Neighborhood structure shape. 3 along all dimension except along the columns where it depends on the sampling rate.
        str_shape = [3 for _ in range(len(shape))] + [2*max(sampling)+1]
        
        # Height of the center of the neigborhood structure along the columns.
        half_height = max(sampling)
        
        ## Create neighborhood structure as a binary array that has ones where edges exist.
        structure = np.zeros(str_shape,np.int)
        
        # Downwards-vertical edges.
        slices = [slice(1,2) for _ in range(len(shape))] + [half_height-1]
        structure[slices] = 1
        
        # Diagonal downards edges.
        for i in range(len(shape)):
            slices_p = [slice(1,2) if i != j else slice(0,1) for j in range(len(shape))] + [half_height-sampling[i]]
            slices_m = [slice(1,2) if i != j else slice(2,3) for j in range(len(shape))] + [half_height-sampling[i]]
            
            structure[slices_p] = 1
            structure[slices_m] = 1
            




        #~ MAX_BOUND = np.finfo(np.float32).max
        #~ MAX = MAX_BOUND/128
        # Used as weight for pseudo-infinite edges.
        
        t0_graph = time()
        # time graph construction.
            



        
        
        augmented_shape = shape + [column_height]
        # Graph shape is image shape + one dimension the size of a column.


        
        ### Write number of nodes, maximal distance, number of dimensions, the shape of the nodes in the graph and the shape of the neighborhood structure.
        
        n_nodes = np.prod(augmented_shape)
        n_dimensions = len(augmented_shape)
        
        f_out.write("# n_nodes, column_height, dims, shape_weights, shape_neighborhood\n")
        f_out.write(" ".join([str(n_nodes),str(column_height),str(n_dimensions)] + [str(el) for el in augmented_shape] + [str(el) for el in structure.shape]))
        f_out.write("\n#\n")

        f_out.write("# neighborhood structure\n")
        f_out.write(" ".join([str(el) for el in structure.ravel()]))
        f_out.write("\n#\n")

 
 
 
 
 
        ### Write terminal edge weights into the file.
        
        slice_ = [slice(None) for _ in range(len(distance_map.shape))] + [np.newaxis]

        if hasattr(cost_fun, '__call__'): # Is a function handle
            weights = cost_fun(distance_map, augmented_shape)
            
        elif isinstance(cost_fun,np.ndarray):
            if cost_fun.shape == (max_dist+1,max_dist+1):
                weights = cost_fun[distance_map.astype(np.int)]
            else:
                raise Exception("Cost function has to be a square array with n_levels^2 shape. Provided array had shape: (" + str(cost_fun.shape[0]) + ', ' + str(cost_fun.shape[1]) + ').')
            
        elif cost_fun == 0 or (isinstance(cost_fun,str) and cost_fun.lower() in ['lin','linear']):
            weights = np.abs(np.ones(augmented_shape)*range(column_height) - distance_map[slice_]) # Linear cost.
            
        elif cost_fun == 1 or (isinstance(cost_fun,str) and cost_fun.lower() in ['lin_clipped','linear_clipped','lin clipped','linear clipped']):
            weights = np.minimum(np.abs(np.ones(augmented_shape)*range(column_height) - distance_map[slice_]),clipping_dist) # Linear cost with upper bound.
            
        elif cost_fun == 2 or (isinstance(cost_fun,str) and cost_fun.lower() in ['weighted']):
            weights = np.abs(np.ones(augmented_shape)*range(column_height) - distance_map[slice_])*(max_dist-distance_map[slice_]) # Linear cost weighted by inverse distance. All costs are 0 if predicted distance = max_dist, and slope is maximal when predicted distance is 0.
        
        elif cost_fun == 3 or (isinstance(cost_fun,str) and cost_fun.lower() in ['weighted_asymmetric','weighted_asym']):
             # Linear cost weighted by inverse distance. All costs are 0 if predicted distance = max_dist, and slope is maximal when predicted distance is 0. Costs are larger for distances larger than prediction (e.g: c(0) = 4, c(1) = 2, c(2) = 0, c(3) = 4, c(4) = 8...)
            weights = np.ones(augmented_shape)*range(column_height) - distance_map[slice_] # Linear cost with upper bound.
            weights[weights<0] = weights[weights<0]*-0.5
            weights = weights*(max_dist-distance_map[slice_]) # Linear cost with upper bound.
        
        # Weights between source/sink and nodes are determined based on the cost for selecting a node.
        slice_ = [slice(None) for _ in range(len(shape))] + [slice(0,1)]
        weights_diff = np.concatenate((weights[slice_], np.diff(weights,axis=-1)), axis=-1)
        
        f_out.write("# weights list\n")
        f_out.write(" ".join([str(el) for el in weights_diff.ravel()]))
        f_out.write("\n#\n")
            
            
            

    f_out.close()
    return augmented_shape
    
    
def call_graph_cut(input_file, output_file, prog, verbose=False):
    '''
    Calls the C implementation to solve the graph-cut problem.
    
    # Parameters
    ============
    
    input_file: string
        Path where the graph was written by 'prepare_problem'.
        
    output_file: string
        Path where to write the output of the graph-cut procedure.
        
    prog: string
        Path to the C executable.
        
    
    '''
    if verbose:
        returned = call([prog, input_file, output_file])
    else:
        with open(os.devnull,'w') as FNULL:
            returned = call([prog, input_file, output_file], stdout=FNULL, stderr=STDOUT)
    
    
def get_graph_cut_output(output_file, shape):
    '''
    Reads the output from the graph-cut procedure from a file and returns a
    ndarray containing the smoothed distance map.
    
    # Parameters
    ============
    
    output_file: string
        Path to the file to be read.
        
    shape: tuple or list
        Shape of the original distance map.
        
        
    # Example
    =========
    
    path_to_my_file = '/my/path/to/output.txt'
    
    shape = [48,48,48]
    
    my_distance_map = get_graph_cut_output(path_to_my_file, shape)
    
    import pylab as pl
    
    pl.imshow(my_distance_map[:,:,24], interpolation='nearest')
    
    '''
    
    with open(output_file,'rb') as f:
        array = np.fromfile(f,dtype=np.bool,sep=' ').reshape(shape)
        
    return np.sum(array,axis=-1)-1
    
    
    
def reconstruct_surface(image, path_graph, path_output, C_prog, max_dist=None, sampling=None, overwrite=False, clipping_dist=4, cost_fun=None, verbose=False):
    
    shape = prepare_problem(image, path_graph, max_dist=max_dist, sampling=sampling, overwrite=overwrite, clipping_dist=clipping_dist, cost_fun=cost_fun)
    
    call_graph_cut(path_graph, path_output, C_prog, verbose=verbose)
    
    return get_graph_cut_output(path_output, shape)
    
    
def score(image, ground_truth, score='L1'):
    '''
    # Parameters
    ============
    
    image: numpy.ndarray
        Array containing the regressed distance.
    
    ground_truth: numpy.ndarray
        Array containing the true distance.
        
    score: string
        One of 'L1', 'L2', 'VI'.
        
    '''
    
    if str.upper(score) == 'L1':
        
        return np.mean(np.abs(image-ground_truth))
        
    elif str.upper(score) == 'L2':
        
        return np.mean((image-ground_truth)**2)
        
    elif str.upper(score) == 'VI':
        
        score_ = 0
        n = np.size(image)
        for val in range(0,int(np.maximum(np.max(image)+1,np.max(ground_truth)+1))):
            idx = image==val
            idy = ground_truth==val
            
            nx = np.sum(idx)*1./n
            ny = np.sum(idy)*1./n
            
            rxy = np.sum(np.logical_and(idx,idy))*1./n
            
            if rxy > 0:
                score_ -= rxy*(np.log(rxy/nx)+np.log(rxy/ny))
            
        return score_
    
    elif str.lower(score) in ['percentage', 'perc', 'p']:
        
        return vdm.calculate_perc_of_correct(ground_truth.astype(np.int), image.astype(np.int))*100    
    else:
        
        raise Exception("Not recognized")
   
        
def best_thresh(image, ground_truth, max_dist=None, score_func='L1'):
    
    if max_dist == None:
        max_dist = int(np.maximum(  np.max(image), np.max(ground_truth) ) )
    
    thresholds = range(max_dist-1)
    
    scores = [score(image<threshold,ground_truth<1,score_func) for threshold in thresholds]
    
    if score_func in ['percentage', 'perc', 'p']:
        return np.max(scores), thresholds[np.argmax(scores)]
    else:
        return np.min(scores), thresholds[np.argmin(scores)]


def plot_histogram(values, ground_truth, max_dist):
    import matplotlib.pyplot as plt
    plt.hist2d(ground_truth, values, (max_dist+1, max_dist+1), cmap=plt.cm.jet, norm=mpl.colors.LogNorm())
    plt.xlabel('ground truth')
    plt.ylabel('prediction')
    plt.title('2d histogram of distance to membrane results')
    plt.show()

def print_scores(ground_truth, noisy_distance, smoothed_distance):
    
    #--------------------------------------------------------------------------
    # Noisy distance scores
    #--------------------------------------------------------------------------
    score_pred, T_pred = best_thresh(noisy_distance, ground_truth, score_func='L1')
    VI_pred, T_pred_VI = best_thresh(noisy_distance, ground_truth, score_func='VI')
    perc_pred, T_pred_perc = best_thresh(noisy_distance, ground_truth, score_func='percentage')
    
    VI_pred_dist = score(noisy_distance, ground_truth,'VI')
    L1_err_pred = score(noisy_distance, ground_truth,'L1')
    L2_err_pred = score(noisy_distance, ground_truth,'L2')
    perc_pred = score(noisy_distance, ground_truth,'percentage')
    
    
    

    print "\n\n\t\t----------------------"
    print '\033[1m' + "\t\t\tSCORES" + '\033[0m'
    print "\t\t----------------------\n"
    print "CNN prediction:\n"
    print "\t-Segmentation:\n"
    print "\t    -L1:"
    print "\t        Best threshold: %d" % T_pred
    print "\t        Error: %.5f" % score_pred
    print "\t    -Percentage correct:"
    print "\t        Best threshold: %d" % T_pred_perc
    print "\t        %% correct: %.5f" % perc_pred
    print "\t    -Variation of Information:"
    print "\t        Best threshold: %d"% T_pred_VI
    print "\t        Error: %.5f\n" % VI_pred
    print "\t-Distance map:\n"
    print "\t    -L1 error: %.3f" % L1_err_pred
    print "\t    -L2 error: %.1f" % L2_err_pred
    print "\t    -Percentage correct: %.2f%%" % perc_pred
    print "\t    -VI score: %.3f\n\n" % VI_pred_dist    
    
    
    
    
    score_smoothed, T_smoothed = best_thresh(smoothed_distance, ground_truth, score_func='L1')
    VI_smoothed, T_smoothed_VI = best_thresh(smoothed_distance, ground_truth, score_func='VI')
    perc_smoothed, T_smoothed_perc = best_thresh(smoothed_distance, ground_truth, score_func='percentage')
    
    VI_smoothed_dist = score(smoothed_distance, ground_truth,'VI')
    L1_err_smoothed = score(smoothed_distance, ground_truth,'L1')
    L2_err_smoothed = score(smoothed_distance, ground_truth,'L2')
    perc_smoothed = score(smoothed_distance, ground_truth,'percentage')
  
  
    print "Smoothed prediction:\n"
    print "\t-Segmentation:\n"
    print "\t    -L1:"
    print "\t        Best threshold: %d" % T_smoothed
    print "\t        Error: %.5f" % score_smoothed
    print "\t    -Percentage correct:"
    print "\t        Best threshold: %d" % T_smoothed_perc
    print "\t        %% correct: %.5f" % perc_smoothed
    print "\t    -Variation of Information:"
    print "\t        Best threshold: %d"% T_smoothed_VI
    print "\t        Error: %.5f\n" % VI_smoothed
    print "\t-Distance map:\n"
    print "\t    -L1 error: %.3f" % L1_err_smoothed
    print "\t    -L2 error: %.1f" % L2_err_smoothed
    print "\t    -Percentage correct: %.2f%%" % perc_smoothed
    print "\t    -VI score: %.3f\n" % VI_smoothed_dist 
        
    
def test(image, path_graph, path_output, C_prog, max_dist=None):
    
    from time import time
    times = []
    sizes = [16,24,28,32,36,40,44]
    for size in sizes:
        
        t0 = time()
        
        res = reconstruct_surface(image[:size,:size,:size], path_graph, path_output, C_prog, overwrite=True, max_dist=None)
    
        t1 = time()
        times.append(t1-t0)
        
    import pylab as pl
    
    pl.plot(sizes,times,'o')
