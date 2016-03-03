import numpy as np
import pylab as pl
import h5py
import os
import sys

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


sys.path.append(os.path.join(ROOT_PATH,'src','python'))

import validate_distance_maps as vdm
import surface_reconstruction as sr
reload(sr)

if __name__ == "__main__":
    
    # -------------------------------------------------------------------------
    # 0. Parse parameters
    # -------------------------------------------------------------------------
    CUBE_SIZE = None
    
    if len(sys.argv) > 2:
        inputfilename = sys.argv[1]
        tmp_files = sys.argv[2]
    else:
        raise Exception("Location of .h5 file required as first argument and path to folder where to create temp files as second argument.")
    
    if len(sys.argv) > 3:
        CUBE_SIZE = int(sys.argv[3])
    
    if len(sys.argv) > 4:
        PLOTTING = np.bool(int(sys.argv[4]))
    else:
        PLOTTING = True
    
    tmp_file1 = os.path.join(tmp_files,'tmp.txt')
    tmp_file2 = os.path.join(tmp_files,'tmp_output.txt')
    prog = os.path.join(ROOT_PATH,'src','cpp','graph_cut')
    
    cube_size = 24 if CUBE_SIZE == None else CUBE_SIZE
    
    
    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    
    f = h5py.File(inputfilename, 'r')
    
    predicted_distances =  f['inference_results'].value
    true_distances = f['labels'].value
    
    f.close()
    
    
    max_dist = 15
    
    start = 20
    
    # Crop a cube of given edge.
    pos = list(np.random.randint(0,256-cube_size,2)) + list(np.random.randint(0,np.maximum(128-cube_size,1),1)) 
    
    
    cube_slice = [slice(pos[0],pos[0]+cube_size), slice(pos[1],pos[1]+cube_size),slice(pos[2],pos[2]+cube_size)]
    
    out = sr.reconstruct_surface(predicted_distances[cube_slice], tmp_file1,tmp_file2,prog,overwrite=True, max_dist=max_dist, sampling = [1,1,2])


    
    sr.print_scores(true_distances[cube_slice], predicted_distances[cube_slice], out)
    

        

        


    if PLOTTING:
        
        # Plot 3 slices throughout the cube.
        
        row_length = 3
        sl = [int(s) for s in np.linspace(0,cube_size,5)[1:-1]]
        cmap = pl.get_cmap('YlGnBu')
        cmap.set_under([0.7,0.95,0.5])
        fig1 = pl.figure()
        for i in range(3):
            
            pl.subplot(3,row_length,i*row_length+1)
            
            if i == 0:
                pl.title('Noisy Distance')
            if len(np.shape(out)) == 3:
                slices_cube = cube_slice[:-1] + [pos[2]+sl[i]]
            else:
                slices_cube = [None,None]
            pl.imshow(np.squeeze(predicted_distances[slices_cube]), vmin=0.5, vmax=15,interpolation='nearest',cmap=cmap)
            pl.xticks([])
            pl.yticks([])
            pl.ylabel('Slice ' + str(sl[i]))
            
            pl.subplot(3,row_length,i*row_length+2)
            
            if i == 0:
                pl.title('Smoothed Distance')
            if len(np.shape(out)) == 3:
                slices = [slice(None),slice(None),sl[i]]
            else:
                slices = [None,None]
            pl.imshow(np.squeeze(out[slices]), vmin=0.5, vmax=15,interpolation='nearest',cmap=cmap)
            pl.xticks([])
            pl.yticks([])
            
            
            pl.subplot(3,row_length,i*row_length+3)
            
            if i == 0:
                pl.title('Ground Truth')
            if len(np.shape(out)) == 3:
                slices_cube = cube_slice[:-1] + [pos[2]+sl[i]]
            else:
                slices_cube = [None,None]
            pl.imshow(np.squeeze(true_distances[slices_cube]), vmin=0.5, vmax=15,interpolation='nearest',cmap=cmap)
            pl.xticks([])
            pl.yticks([])
        pl.suptitle(str(cube_size) + ' x ' + str(cube_size) + ' slices comparison')
            
            
        
        
        
        
        # Plot the 2D histograms
            
        pl.figure()
        prediction = predicted_distances[cube_slice].flatten()
        results = out.flatten()
        labels = true_distances[cube_slice].flatten()
        pl.subplot(1,2,1)
        pl.hist2d(labels, prediction, (max_dist+1, max_dist+1), cmap=pl.cm.jet, norm=pl.mpl.colors.LogNorm())
        pl.xlabel('ground truth')
        pl.ylabel('noisy distance map')
        pl.title('2d histogram of noisy distance')
        
        pl.subplot(1,2,2)
        pl.hist2d(labels, results, (15, 15), cmap=pl.cm.jet, norm=pl.mpl.colors.LogNorm())
        pl.xlabel('ground truth')
        pl.ylabel('smoothed distance map')
        pl.title('2d histogram of smoothed distance')
        pl.show()
        
                
        
