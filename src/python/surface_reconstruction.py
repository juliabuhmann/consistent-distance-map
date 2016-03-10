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
reload(vdm)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



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
    
    n_dims = len(shape)
    
    
    if sampling != None:
        assert len(sampling) == len(shape)
        assert min(sampling) >= 1


    if n_dims == 2:
        return_2D = True
        shape = shape + [1]
        if sampling != None:
            sampling = sampling + [1]
        distance_map = distance_map.reshape(shape)
        n_dims += 1
    else:
        return_2D = False

    if max_dist == None:
        max_dist = int(np.max(distance_map))
        print "Max distance not defined. Using " + str(max_dist) + "." 
    
    if cost_fun is None:
        cost_fun='lin_clipped'
        
    column_height = max_dist+1 # ranges from 0 to max_dist

    if os.path.exists(path_output) and not overwrite:
        raise Exception(path_output + ' already exists. Switch "overwrite" argument to True to replace it.')


    with open(path_output,'wb') as f_out:
        
        if sampling == None: # If not defined, assume isotropic sampling.
            sampling = [1 for _ in range(n_dims+1)]
            
        
        # Neighborhood structure shape. 3 along all dimension except along the columns where it depends on the sampling rate.
        str_shape = [3 for _ in range(n_dims)] + [2*max(sampling)+1]
        
        # Height of the center of the neigborhood structure along the columns.
        half_height = max(sampling)
        
        ## Create neighborhood structure as a binary array that has ones where edges exist.
        structure = np.zeros(str_shape,np.int)
        
        # Downwards-vertical edges.
        slices = [slice(1,2) for _ in range(n_dims)] + [half_height-1]
        structure[slices] = 1
        
        # Diagonal downards edges.
        for i in range(n_dims):
            slices_p = [slice(1,2) if i != j else slice(0,1) for j in range(n_dims)] + [half_height-sampling[i]]
            slices_m = [slice(1,2) if i != j else slice(2,3) for j in range(n_dims)] + [half_height-sampling[i]]
            
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
        
        weights = compute_weights(distance_map, max_dist, cost_fun=cost_fun, clipping_dist=clipping_dist)
        
        
        # Weights between source/sink and nodes are determined based on the cost for selecting a node.
        slice_ = [slice(None) for _ in range(n_dims)] + [slice(0,1)]
        
        weights_diff = np.concatenate((weights[slice_], np.diff(weights,axis=-1)), axis=-1)
        #print weights_diff
        
        f_out.write("# weights list\n")
        f_out.write(" ".join([str(el) for el in weights_diff.ravel()]))
        f_out.write("\n#\n")
            
            
            

    f_out.close()
    return augmented_shape





def prepare_problem_VCE(distance_map, path_output, max_dist=None, sampling=None, overwrite=False, cost_fun='lin_clipped',clipping_dist=4, binaries=None):
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
    
    n_dims = len(shape)
    
    
    if distance_map.dtype != np.int:
        print bcolors.WARNING + "Distance map is not integer. Rounding values." + bcolors.ENDC
        distance_map = np.round(distance_map).astype(np.int)
    
    if np.min(distance_map) < 0:
        print bcolors.WARNING + "Negative values found in distance map. Clipping them to 0." + bcolors.ENDC
        distance_map = np.maximum(distance_map,0)
        
    if max_dist == None:
        max_dist = int(np.max(distance_map))
        print bcolors.WARNING + "Max distance not defined. Using " + str(max_dist) + "."  + bcolors.ENDC
    
    if np.min(distance_map) < 0:
        print bcolors.WARNING + "Values over maximal distance (" + str(max_dist) + ") found. Clipping them to " + str(max_dist) + "." + bcolors.ENDC
        distance_map = np.maximum(distance_map,0)
        
    
    if sampling != None:
        assert len(sampling) == len(shape)
        assert min(sampling) >= 1


    if n_dims == 2:
        return_2D = True
        shape = shape + [1]
        if sampling != None:
            sampling = sampling + [1]
        n_dims += 1
    else:
        return_2D = False

    
    if cost_fun is None:
        cost_fun='lin_clipped'
    
    
    if binaries is None:
        pass#binaries=np.zeros([max_dist+1]*4)
    elif type(binaries) == np.ndarray:
        assert all(np.array(binaries.shape) == [max_dist+1]*4)
    elif isinstance(binaries,str):
        binaries = get_binaries_mat(max_dist,binaries)
    
    
    column_height = max_dist+1 # ranges from 0 to max_dist

    if os.path.exists(path_output) and not overwrite:
        raise Exception(path_output + ' already exists. Switch "overwrite" argument to True to replace it.')


    with open(path_output,'wb') as f_out:
        
        if sampling == None: # If not defined, assume isotropic sampling.
            sampling = [1 for _ in range(n_dims+1)]
            
        
        # Neighborhood structure shape. 3 along all dimension except along the columns where it depends on the sampling rate.
        str_shape = [3 for _ in range(n_dims)] + [2*max(sampling)+1]
        
        # Height of the center of the neigborhood structure along the columns.
        half_height = max(sampling)
        
        ## Create neighborhood structure as a binary array that has ones where edges exist.
        structure = np.zeros(str_shape,np.int)
        
        # Downwards-vertical edges.
        slices = [slice(1,2) for _ in range(n_dims)] + [half_height-1]
        structure[slices] = 1
        
        # Diagonal downards edges.
        for i in range(n_dims):
            slices_p = [slice(1,2) if i != j else slice(0,1) for j in range(n_dims)] + [half_height-sampling[i]]
            slices_m = [slice(1,2) if i != j else slice(2,3) for j in range(n_dims)] + [half_height-sampling[i]]
            
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
        
        f_out.write("# n_nodes, max_dist, dims, shape_weights, shape_neighborhood\n")
        
        f_out.write(" ".join([str(n_nodes),str(max_dist),str(n_dimensions)] + [str(el) for el in augmented_shape] + [str(el) for el in structure.shape]))
        f_out.write("\n#\n")

        f_out.write("# neighborhood structure\n")
        f_out.write(" ".join([str(el) for el in structure.ravel()]))
        f_out.write("\n#\n")


 
 
 
 
        ### Write terminal edge weights into the file.
        
        unaries = compute_weights(np.array(range(max_dist+1)), max_dist, cost_fun=cost_fun, clipping_dist=clipping_dist)
        
        unaries = np.concatenate((unaries[:,0:1], np.diff(unaries,axis=-1)), axis=-1)
        
        
        f_out.write("# unaries list\n")
        f_out.write(" ".join([str(el) for el in unaries.ravel()]))
        f_out.write("\n#\n")
 
 
 
        ### Write terminal edge weights into the file.
        if not binaries is None:
            if isinstance(binaries,str):
                binaries = get_binaries_mat(max_dist, binaries)
            
            
            f_out.write("binaries list\n")
            f_out.write(" ".join([str(el) for el in binaries.ravel()]))
            f_out.write("\n#\n")
            
            
        f_out.write("# predicted distance map\n")
        f_out.write(" ".join([str(el) for el in distance_map.ravel()]))
        f_out.write("\n#\n")    

    f_out.close()
    print "\n"
    return augmented_shape



def compute_weights(distance_map, max_dist, cost_fun='linear', clipping_dist=4):
    
    shape = np.shape(distance_map)
    n_dims = len(shape)
    column_height = max_dist+1
    augmented_shape = list(shape) + [column_height]
    
    if len(distance_map.shape) < n_dims: # 2D data
        slice_ = [slice(None) for _ in range(n_dims-1)] + [np.newaxis,np.newaxis]
    else:
        slice_ = [slice(None) for _ in range(len(shape))] + [np.newaxis]

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
    else:
        raise Exception("Cost function was not understood.")
        
    return weights





def get_binaries_mat(max_dist, binaries, *args):
    
    col_height = max_dist+1
    if binaries == 'zeros':
        return np.zeros([col_height]*4,np.float)
    elif binaries == 'linear':
        if len(args):
            slope = args[0]
        else:
            slope = 0.75
        return np.array([[slope*np.abs(np.array(range(col_height),np.float)[:,np.newaxis]-np.array(range(col_height),np.float)[np.newaxis,:]) for _ in range(col_height)] for __ in range(col_height)])
    if binaries == 'zeros':
        return np.zeros([col_height]*4,np.float)



    
    
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
        
    return np.squeeze(np.sum(array,axis=-1)-1)
    
    
    
def reconstruct_surface(image, path_graph, path_output, C_prog, max_dist=None, sampling=None, overwrite=False, clipping_dist=4, cost_fun=None, verbose=False, binaries=None):
    
    if binaries is None:
        shape = prepare_problem(image, path_graph, max_dist=max_dist, sampling=sampling, overwrite=overwrite, clipping_dist=clipping_dist, cost_fun=cost_fun)
    else:
        shape = prepare_problem_VCE(image, path_graph, max_dist=max_dist, sampling=sampling, overwrite=overwrite, clipping_dist=clipping_dist, cost_fun=cost_fun, binaries=binaries)
    
    call_graph_cut(path_graph, path_output, C_prog, verbose=verbose)
    
    return get_graph_cut_output(path_output, shape)
    
    
    
    
    
def reconstruct_surface_VCE(image, path_graph, path_output, C_prog, max_dist=None, sampling=None, overwrite=False, clipping_dist=4, cost_fun=None, binaries=None, verbose=False):
    
    shape = prepare_problem_VCE(image, path_graph, max_dist=max_dist, sampling=sampling, overwrite=overwrite, clipping_dist=clipping_dist, cost_fun=cost_fun, binaries=binaries)
    
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
        One of 'L1', 'L2', 'VI', 'CC_VI'.
        
    '''
    
    if str.upper(score) == 'L1':
        
        return np.mean(np.abs(image-ground_truth))
        
    elif str.upper(score) == 'L2':
        
        return np.mean((image-ground_truth)**2)
        
    elif str.upper(score) == 'VI':
        
        score_ = 0
        n = float(np.size(image))
        
        for val in np.unique(image):
            for val2 in np.unique(ground_truth):
                idx = image==val
                idy = ground_truth==val2
                
                nx = np.sum(idx)/n
                ny = np.sum(idy)/n
                
                rxy = np.sum(np.logical_and(idx,idy))/n
                
                if rxy > 0:
                    score_ -= rxy*(np.log(rxy/nx)+np.log(rxy/ny))
            
        return score_
        
    elif str.upper(score) == 'CC_VI':
        im1 = _get_segmentation(np.invert(image.astype(np.bool)))[0]
        im2 = _get_segmentation(np.invert(ground_truth.astype(np.bool)))[0]
        
        return _varinfo(im1,im2)
    
    elif str.lower(score) in ['percentage', 'perc', 'p']:
        
        return vdm.calculate_perc_of_correct(ground_truth.astype(np.int), image.astype(np.int))*100    
    else:
        
        raise Exception("Not recognized")

def _get_segmentation(binary_image):
    
    return ni.label(binary_image)



def _conn_comp_VI(binary_image1, binary_image2,edge_image=True):
    # Replaced by _varinfo
    
    
    
    if edge_image:
        seg1, n1 = _get_segmentation(np.invert(binary_image1))
        seg2, n2 = _get_segmentation(np.invert(binary_image2))
    else:
        seg1, n1 = _get_segmentation(binary_image1)
        seg2, n2 = _get_segmentation(binary_image2)
    
    score_ = 0.
    n = np.size(binary_image1)
    for val in range(1,n1+1):
        for val2 in range(1,n2+1):
            id1 = seg1==val
            id2 = seg2==val2
            
            nx = np.sum(id1)*1.
            ny = np.sum(id2)*1.
            
            #rxy = np.sum(np.logical_and(id1,id2))*1./n
            rxyn = np.sum(np.logical_and(id1,id2))*1.
            
            if rxyn > 0.:
                score_ -= rxyn*(np.log(rxyn/nx)+np.log(rxyn/ny))/n
            #if stop:
                #print 'n: %d, nx: %f, ny: %f, rxy: %f, score_diff: %f' % (n,nx,ny,rxy,rxy*(np.log(rxy/nx)+np.log(rxy/ny)))
    
                    
    return score_    
  
  
def _entropy(label):
    N = np.size(label)
    #~ N = np.sum(label>0)
    k = [el for el in np.unique(label)]# if el != 0]
    #~ k = [el for el in np.unique(label) if el != 0]
    H = 0.
      
      
    for i in k:
        pk = float(np.sum(i == label))/N
        H -= pk*np.log(pk)
        if np.isnan(H):
            raise Exception()
    return H
    
def _varinfo(label1,label2):
    
    h1 = _entropy(label1)
    h2 = _entropy(label2)

    i12 = _mutualinfo(label1,label2)
    
    return h1 + h2 - 2*i12

def _mutualinfo(label1,label2): 
    
    N = float(np.size(label1))
    #~ N = float(np.sum(label1+label2>0))
    k1 = [el for el in np.unique(label1)]# if el != 0]
    k2 = [el for el in np.unique(label2)]# if el != 0]
    #~ k1 = [el for el in np.unique(label1) if el != 0]
    #~ k2 = [el for el in np.unique(label2) if el != 0]
    I = 0


    for i in k1:
        # loop over the unique elements of L2
        for j in k2:
            # the mutual probability of two classification indices occurring in
            # L1 and L2
            pij = np.sum((label1 == i)*(label2 == j))/N
            # the probability of a given classification index occurring in L1
            pi = np.sum(label1 == i)/N
            # the probability of a given classification index occurring in L2
            pj = np.sum(label2 == j)/N
            if pij > 0:
                I += pij*np.log(pij/(pi*pj))
            
    return I
        
def best_thresh(image, ground_truth, max_dist=None, score_func='L1'):
    
    if max_dist == None:
        max_dist = int(np.maximum(  np.max(image), np.max(ground_truth) ) )
    
    thresholds = range(1,max_dist-1)
    
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
    a = np.copy(ground_truth)
    b = np.copy(noisy_distance)
    c = np.copy(smoothed_distance)
    score_pred, T_pred = best_thresh(noisy_distance, ground_truth, score_func='L1')
    VI_pred, T_pred_VI = best_thresh(noisy_distance, ground_truth, score_func='VI')
    CC_VI_pred, T_pred_CC_VI = best_thresh(noisy_distance, ground_truth, score_func='CC_VI')
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
    print "\t        Error: %.5f" % VI_pred
    print "\t    -Variation of Information on Connected Components:"
    print "\t        Best threshold: %d"% T_pred_CC_VI
    print "\t        Error: %.5f\n" % CC_VI_pred
    print "\t-Distance map:\n"
    print "\t    -L1 error: %.3f" % L1_err_pred
    print "\t    -L2 error: %.1f" % L2_err_pred
    print "\t    -Percentage correct: %.2f%%" % perc_pred
    print "\t    -VI score: %.3f\n\n" % VI_pred_dist    
    
    
    
    
    score_smoothed, T_smoothed = best_thresh(smoothed_distance, ground_truth, score_func='L1')
    VI_smoothed, T_smoothed_VI = best_thresh(smoothed_distance, ground_truth, score_func='VI')
    CC_VI_smoothed, T_smoothed_CC_VI = best_thresh(smoothed_distance, ground_truth, score_func='CC_VI')
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
    print "\t        Error: %.5f" % VI_smoothed
    print "\t    -Variation of Information on Connected Components:"
    print "\t        Best threshold: %d"% T_smoothed_CC_VI
    print "\t        Error: %.5f\n" % CC_VI_smoothed
    print "\t-Distance map:\n"
    print "\t    -L1 error: %.3f" % L1_err_smoothed
    print "\t    -L2 error: %.1f" % L2_err_smoothed
    print "\t    -Percentage correct: %.2f%%" % perc_smoothed
    print "\t    -VI score: %.3f\n" % VI_smoothed_dist 
        
    print     np.all(a == ground_truth)
    print     np.all(b == noisy_distance)
    print     np.all(c == smoothed_distance)
    
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






def solve_via_ILP(weights, max_gradient=1, enforce_minimum=False, num_cores=None, solve_relaxed_problem=False):
    try:
        if os.path.exists('/data/owncloud/MinCutForDistance/pysurfrec/build/python'):
            sys.path.append('/data/owncloud/MinCutForDistance/pysurfrec/build/python')
        import surfrec
            
    except ImportError, e:
        print "Module pysurfrec not found. Make sure you installed it correctly and that it is in your python path."
        raise e
    
    
    
    
    
    num_levels = weights.shape[-1]
    max_dist = num_levels-1
    
    shape = weights.shape[:-1]
    full_shape = weights.shape
    n_dims = len(shape)
    
    if isinstance(max_gradient,int):
        max_gradient = [max_gradient for _ in range(n_dims)]
    
    num_nodes = int(np.prod(shape))
    num_edges = int(sum([np.prod(shape[:i])*np.prod(shape[i+1:])*(shape[i]-1) for i in range(n_dims) ]))

    #surfrec.setLogLevel(surfrec.LogLevel.All)

    # "Instantiating solver"
    print "Max gradient:", max_gradient
    s = surfrec.IlpSolver(num_nodes, num_nodes - 1, num_levels, max_gradient[0])

    p = surfrec.IlpSolverParameters()
    p.enforce_zero_minimum = enforce_minimum
    p.num_neighbors = 2*n_dims
    p.solve_relaxed_problem = solve_relaxed_problem
    
    
    if not num_cores is None:
        p.num_threads = num_cores
        





    #print "Adding nodes"
    first = s.add_nodes(num_nodes)
    nodes = np.reshape(range(first, first + num_nodes), shape)

    #print "Adding edges"
    for i in range(num_nodes - 1):
        
        coords = np.unravel_index(i,shape)
        #print i, "-->", coords
        for dim in range(n_dims):
            neigh = coords + np.array([1 if j == dim else 0 for j in range(n_dims)])
            #print coords, neigh, shape
            if all(neigh < shape) and all(neigh >= 0):
                i_neigh = np.ravel_multi_index(neigh,shape)
                #print str(i) + ": Linking level " + str(coords) + " to " +str(neigh) + "."
                s.add_edge(nodes[coords], nodes[tuple(neigh)], max_gradient[dim])
                
                
    print "Adding level costs for the " + str(num_nodes) + " nodes."
    for i in range(num_nodes):
        coords = list(np.unravel_index(i,shape))
        costs = surfrec.ColumnCosts(num_levels)
        for j in range(num_levels):
            costs[j] = weights[tuple(coords + [j])];
        s.set_level_costs(nodes[tuple(coords)], costs);



    print "Solving"
    t0 = time()
    value = s.min_surface(p)
    print "Solution found in", time()-t0, "seconds."



    return np.reshape([ s.level(n) for n in nodes.flatten() ],shape)
