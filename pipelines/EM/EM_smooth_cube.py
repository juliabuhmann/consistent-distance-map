import numpy as np
import pylab as pl
import h5py
import os
import sys
from time import time
from tifffile import imsave
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_DATA = '/Users/abouchar/ownCloud/ProjectRegSeg/data/EM_cubes'
ROOT_DATA = '/data/owncloud/ProjectRegSeg/data/EM_cubes'

sys.path.append(os.path.join(ROOT_PATH,os.pardir,os.pardir,'src','python'))

import results_evaluation as evaluate
import surface_reconstruction as sr
reload(sr)

if __name__ == "__main__":
    
    
    cost_functions = {'linear':None,'linear_clipped_2':2,'linear_clipped_4':4,'linear_clipped_6':6,'weighted':None,'weighted_asymmetric':None,'prior':'feature_collection_training_cube1_softmax.h5'}
    
    times = dict((key, []) for key in cost_functions)
    
    
    # -------------------------------------------------------------------------
    # 0. Parse parameters
    # -------------------------------------------------------------------------
    CUBE_SIZE = None
    
    #if len(sys.argv) > 2:
    inputfilename = os.path.join(ROOT_DATA,'feature_collection_training_cube0_softmax.h5')
    tmp_files = '/Users/abouchar/Desktop/dump'
    tmp_files = '/data/dump'
    #~ else:
        #~ raise Exception("Location of .h5 file required as first argument and path to folder where to create temp files as second argument.")
    
    
    
    PLOTTING = False
    
    
    tmp_file1 = os.path.join(tmp_files,'tmp.txt')
    tmp_file2 = os.path.join(tmp_files,'tmp_output.txt')
    
    prog = os.path.join(ROOT_PATH,os.pardir,os.pardir,'src','cpp','graph_cut')
    
    cube_size = 64
    
    
    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    
    f = h5py.File(inputfilename, 'r')
    
    predicted_distances =  f['inference_results'].value
    true_distances = f['labels'].value
    predicted_distances = np.minimum(predicted_distances,np.max(true_distances))
    f.close()
    
    
    max_dist = 15
    
    start = 20
    # Crop a cube of given edge.
    
    subcubes = [[slice(x,x+64),slice(y,y+64),slice(z,z+64)] for x in range(0,256,64) for y in range(0,256,64) for z in range(0,128,64)]
    
    for cube_ind, cube in enumerate(subcubes):
        print "\nStarting with subcube " + str(cube_ind+1) + "/" + str(len(subcubes)) + ".\n"
        for cost_function, params in cost_functions.iteritems():
            print "Using " + cost_function + " unaries.\n"
            np.random.seed(555)
            
            output = os.path.join(ROOT_DATA,'smoothed',cost_function,'cube0_' + str(cube_ind) + '.tif')
            output_flag = os.path.join(ROOT_DATA,'smoothed',cost_function,'cube0_' + str(cube_ind) + '_start.tif')
            
            open(output_flag, 'a').close()
            
            if os.path.isfile(output) or os.path.isfile(output_flag):
                continue
            
            if params == None:
                t0 = time()
                out = sr.reconstruct_surface_VCE(predicted_distances[cube], tmp_file1,tmp_file2,prog,overwrite=True, max_dist=max_dist, sampling = [1,1,2], cost_fun=cost_function, verbose=True)
                t1 = time()
                
                
                
            elif isinstance(params,int):
                t0 = time()
                out = sr.reconstruct_surface_VCE(predicted_distances[cube], tmp_file1,tmp_file2,prog,overwrite=True, max_dist=max_dist, sampling = [1,1,2], clipping_dist=params, cost_fun=cost_function[:-2], verbose=True)
                t1 = time()
            elif isinstance(params,str):
                
                f = h5py.File(os.path.join(ROOT_DATA,params), 'r')
                predicted_distances_prior =  f['inference_results'].value
                true_distances_prior = f['labels'].value
                f.close()
                
                predicted_distances_prior = np.minimum(predicted_distances_prior,np.max(true_distances))
                prior = pl.hist2d(predicted_distances_prior.flatten(), true_distances_prior.flatten(), range=[[-0.5,max_dist+0.5],[-0.5,max_dist+0.5]],bins=np.linspace(-0.5,max_dist + 0.5,max_dist+2))
                prior = np.exp(-(prior[0] / prior[0].sum(axis=1)[:,np.newaxis].astype(np.float) ))*1000
                pl.close()
                t0 = time()
                out = sr.reconstruct_surface_VCE(predicted_distances[cube], tmp_file1,tmp_file2,prog,overwrite=True, max_dist=max_dist, sampling = [1,1,2], clipping_dist=params, cost_fun=prior, verbose=True)
                t1 = time()
                        
            imsave(output,out.astype(np.int32),compress=1)
            os.remove(output_flat)
            times[cost_function].append(t1-t0)
     
