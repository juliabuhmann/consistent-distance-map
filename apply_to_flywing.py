import numpy as np
import pylab as pl
import os
import sys
from tifffile import imread

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


sys.path.append(os.path.join(ROOT_PATH,'src','python'))

import validate_distance_maps as vdm
import surface_reconstruction as sr

def my_cost_function( distance_map, augmented_shape ):
    column_height = augmented_shape[-1]
    max_dist = column_height-1
    slice_ = [slice(None) for _ in range(len(distance_map.shape))] + [np.newaxis]
    return np.abs(np.ones(augmented_shape)*range(int(column_height)) - distance_map[slice_])*distance_map[slice_]

if __name__ == "__main__":
    
    # -------------------------------------------------------------------------
    # 0. Parse parameters
    # -------------------------------------------------------------------------
    SQUARE_SIZE_X = None
    
    if len(sys.argv) > 3:
        fnPredictedDists = sys.argv[1]
        fnTrueDists = sys.argv[2]
        tmp_files = sys.argv[3]
    else:
        raise Exception("Location of TIFF file required as first argument and path to folder where to create temp files as second argument.")
    
    if len(sys.argv) > 4:
        SQUARE_SIZE_X = int(sys.argv[4])
        if len(sys.argv) > 5:
            SQUARE_SIZE_Y = int(sys.argv[5])
        else:
            SQUARE_SIZE_Y = SQUARE_SIZE_X
    
    if len(sys.argv) > 6:
        PLOTTING = np.bool(int(sys.argv[6]))
    else:
        PLOTTING = True
    
    tmp_file1 = os.path.join(tmp_files,'tmp.txt')
    tmp_file2 = os.path.join(tmp_files,'tmp_output.txt')
    prog = os.path.join(ROOT_PATH,'src','cpp','graph_cut')
    
    
    square_size = [24 if SQUARE_SIZE_X == None else SQUARE_SIZE_X, 24 if SQUARE_SIZE_Y == None else SQUARE_SIZE_Y]
    
    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    
    predicted_distances = np.round(imread(fnPredictedDists))
    true_distances = imread(fnTrueDists)

    sizeX = predicted_distances.shape[1]
    sizeY = predicted_distances.shape[0]
    
    max_dist = int(np.maximum( np.max(predicted_distances), np.max(true_distances) ))
    
    # Crop a cube of given edge.
    pos = list(np.random.randint(0,np.maximum(sizeY-square_size[1],1),1)) + list(np.random.randint(0,np.maximum(sizeX-square_size[0],1),1)) 
    
    square_slice = [slice(pos[0],pos[0]+square_size[1]), slice(pos[1],pos[1]+square_size[0])]
    
    # cost_function = 'linear'
    # cost_function = 'linear_clipped'
    # cost_function = 'weighted'
    cost_function = 'weighted_asymmetric'
    # cost_function = my_cost_function
    out = sr.reconstruct_surface(predicted_distances[square_slice], tmp_file1,tmp_file2,prog,overwrite=True, max_dist=max_dist, sampling = [1,1], cost_fun=cost_function)
    
    
    
    
    sr.print_scores(true_distances[square_slice], predicted_distances[square_slice], out)
    
    
    
        

    PLOT_DIFFS = True # Whether to plot additional rows that show the difference between images.
    if PLOTTING:
        
        if not PLOT_DIFFS:
            # Plot results.
            pl.suptitle(str(square_size[0]) + ' x ' + str(square_size[1]) + ' region')
            
            VMIN = 0.5
            VMAX = max_dist
            cmap = pl.get_cmap('YlGnBu') # Pick a colormap: http://matplotlib.org/examples/color/colormaps_reference.html
            cmap.set_under([0.7,0.95,0.5]) # Color for anything below VMIN
            
            row_length = 3
            pl.subplot(1,row_length,1)
            pl.xticks([])
            pl.yticks([])
            
            pl.title('Predicted Distance')
            pl.imshow(np.squeeze(predicted_distances[square_slice]), vmin=VMIN, vmax=VMAX,interpolation='nearest', cmap=cmap)
            
            pl.subplot(1,row_length,2)
            
            pl.title('Smoothed Distance')
            pl.imshow(np.squeeze(out), vmin=VMIN, vmax=VMAX,interpolation='nearest', cmap=cmap)
            pl.xticks([])
            pl.yticks([])
            
            pl.subplot(1,row_length,3)
            
            pl.title('Ground Truth')
            pl.imshow(np.squeeze(true_distances[square_slice]), vmin=VMIN, vmax=VMAX,interpolation='nearest', cmap=cmap)
            pl.xticks([])
            pl.yticks([])
            
        else:
            # Plot results.
            pl.suptitle(str(square_size[0]) + ' x ' + str(square_size[1]) + ' region')
            VMIN = 0.5
            VMAX = max_dist
            cmap = pl.get_cmap('YlGnBu') # Pick a colormap: http://matplotlib.org/examples/color/colormaps_reference.html
            cmap.set_under([0.7,0.95,0.5]) # Color for anything below VMIN
            
            row_length = 3

            
            
            
            
            pl.subplot(3,row_length,1)
            
            pl.title('Predicted Distance')
            pl.imshow(np.squeeze(predicted_distances[square_slice]), vmin=VMIN, vmax=VMAX,interpolation='nearest', cmap=cmap)
            pl.xticks([])
            pl.yticks([])
            pl.ylabel('Results')
            
            pl.subplot(3,row_length,2)
            
            pl.title('Smoothed Distance')
            pl.imshow(np.squeeze(out), vmin=VMIN, vmax=VMAX,interpolation='nearest', cmap=cmap)
            pl.xticks([])
            pl.yticks([])
            
            pl.subplot(3,row_length,3)
            
            pl.title('Ground Truth')
            pl.imshow(np.squeeze(true_distances[square_slice]), vmin=VMIN, vmax=VMAX,interpolation='nearest', cmap=cmap)
            pl.xticks([])
            pl.yticks([])





            pl.subplot(3,row_length,4)
            
            pl.title('Predicted Distance vs Smoothed Distance')
            pl.imshow(np.abs(predicted_distances[square_slice]-out), vmin=VMIN, vmax=VMAX,interpolation='nearest', cmap=cmap)
            pl.xticks([])
            pl.yticks([])
            pl.ylabel('Distance Comparison')
            
            pl.subplot(3,row_length,5)
            
            pl.title('Predicted Distance vs Ground Truth')
            pl.imshow(np.abs(predicted_distances[square_slice]-true_distances[square_slice]), vmin=VMIN, vmax=VMAX,interpolation='nearest', cmap=cmap)
            pl.xticks([])
            pl.yticks([])
            
            pl.subplot(3,row_length,6)
            
            pl.title('Smoothed Distance vs Ground Truth')
            pl.imshow(np.abs(true_distances[square_slice]-out), vmin=VMIN, vmax=VMAX,interpolation='nearest', cmap=cmap)
            pl.xticks([])
            pl.yticks([])
            
            
            
            
            pl.subplot(3,row_length,7)
            pl.title('Predicted Distance vs Smoothed Distance')
            pl.imshow((np.array(predicted_distances[square_slice]>0,np.int)-np.array(out>0,np.int))*(VMAX-VMIN)*0.5+1, vmin=VMIN, vmax=VMAX,interpolation='nearest', cmap=cmap)
            pl.xticks([])
            pl.yticks([])
            pl.ylabel('Segmentation Comparison')
            
            pl.subplot(3,row_length,8)
            
            pl.title('Predicted Distance vs Ground Truth')
            pl.imshow((np.array(predicted_distances[square_slice]>0,np.int)-np.array(true_distances[square_slice]>0,np.int))*(VMAX-VMIN)*0.5+1, vmin=VMIN, vmax=VMAX,interpolation='nearest', cmap=cmap)
            pl.xticks([])
            pl.yticks([])
            
            pl.subplot(3,row_length,9)
            pl.title('Smoothed Distance vs Ground Truth')
            pl.imshow((np.array(true_distances[square_slice]>0,np.int)-np.array(out>0,np.int))*(VMAX-VMIN)*0.5+1, vmin=VMIN, vmax=VMAX,interpolation='nearest', cmap=cmap)
            pl.xticks([])
            pl.yticks([])
            

            
            
            
            
            
        # Plot the 2D histograms
            
        pl.figure()
        prediction = predicted_distances[square_slice].flatten()
        results = out.flatten()
        labels = true_distances[square_slice].flatten()
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
