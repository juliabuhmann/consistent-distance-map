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
    SQUARE_SIZE = None
    
    if len(sys.argv) > 3:
        fnPredictedDists = sys.argv[1]
        fnTrueDists = sys.argv[2]
        tmp_files = sys.argv[3]
    else:
        raise Exception("Location of .h5 file required as first argument and path to folder where to create temp files as second argument.")
    
    if len(sys.argv) > 4:
        SQUARE_SIZE = int(sys.argv[4])
    
    if len(sys.argv) > 5:
        PLOTTING = np.bool(int(sys.argv[5]))
    else:
        PLOTTING = True
    
    tmp_file1 = os.path.join(tmp_files,'tmp.txt')
    tmp_file2 = os.path.join(tmp_files,'tmp_output.txt')
    prog = os.path.join(ROOT_PATH,'src','cpp','graph_cut')
    
    square_size = 24 if SQUARE_SIZE == None else SQUARE_SIZE
    
    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    
    predicted_distances = np.round(imread(fnPredictedDists))
    true_distances = imread(fnTrueDists)

    sizeX = predicted_distances.shape[1]
    sizeY = predicted_distances.shape[0]
    
    max_dist = int(np.maximum( np.max(predicted_distances), np.max(true_distances) ))
    
    # Crop a cube of given edge.
    pos = list(np.random.randint(0,sizeY-square_size,1)) + list(np.random.randint(0,np.maximum(sizeX-square_size,1),1)) 
    
    square_slice = [slice(pos[0],pos[0]+square_size), slice(pos[1],pos[1]+square_size)]
    
    # cost_function = 'linear'
    # cost_function = 'linear_clipped'
    # cost_function = 'weighted'
    cost_function = 'weighted_asymmetric'
    # cost_function = my_cost_function
    out = sr.reconstruct_surface(predicted_distances[square_slice], tmp_file1,tmp_file2,prog,overwrite=True, max_dist=max_dist, sampling = [1,1], cost_fun=cost_function)
    
    score_pred, T_pred = sr.best_thresh(predicted_distances[square_slice], true_distances[square_slice],max_dist, score_func='L1')
    VI_pred, T_pred_VI = sr.best_thresh(predicted_distances[square_slice], true_distances[square_slice],max_dist, score_func='VI')
    
    VI_pred_dist = sr.score(predicted_distances[square_slice], true_distances[square_slice],'VI')
    L1_err_pred = sr.score(predicted_distances[square_slice], true_distances[square_slice],'L1')
    L2_err_pred = sr.score(predicted_distances[square_slice], true_distances[square_slice],'L2')
    perc_pred = vdm.calculate_perc_of_correct(true_distances[square_slice].astype(np.int), predicted_distances[square_slice].astype(np.int))*100
    
    assert L1_err_pred == vdm.calculate_L1(true_distances[square_slice], predicted_distances[square_slice])
    assert L2_err_pred == vdm.calculate_L2(true_distances[square_slice], predicted_distances[square_slice])
    
    score_smoothed, T_smoothed = sr.best_thresh(out, true_distances[square_slice],max_dist, score_func='L1')
    VI_smoothed, T_smoothed_VI = sr.best_thresh(out, true_distances[square_slice],max_dist, score_func='VI')
    
    VI_smoothed_dist = sr.score(out, true_distances[square_slice],'VI')
    L1_err_smoothed = sr.score(out, true_distances[square_slice],'L1')
    L2_err_smoothed = sr.score(out, true_distances[square_slice],'L2')
    perc_smoothed = vdm.calculate_perc_of_correct(true_distances[square_slice].astype(np.int), out.astype(np.int))*100
    
    assert L1_err_smoothed == vdm.calculate_L1(true_distances[square_slice], out)
    assert L2_err_smoothed == vdm.calculate_L2(true_distances[square_slice], out)
    print L1_err_smoothed,'==', vdm.calculate_L1(true_distances[square_slice], out)
    
    print "\n\n\t\t----------------------"
    print '\033[1m' + "\t\t\tSCORES" + '\033[0m'
    print "\t\t----------------------\n"
    print "CNN prediction:\n"
    print "\t-Segmentation:\n"
    print "\t    -L1:"
    print "\t        Best threshold: %d" % T_pred
    print "\t        Error: %.5f" % score_pred
    print "\t    -Variation of Information:"
    print "\t        Best threshold: %d"% T_pred_VI
    print "\t        Error: %.5f\n" % VI_pred
    print "\t-Distance map:\n"
    print "\t    -L1 error: %.3f" % L1_err_pred
    print "\t    -L2 error: %.1f" % L2_err_pred
    print "\t    -Percentage correct: %.2f%%" % perc_pred
    print "\t    -VI score: %.3f\n\n" % VI_pred_dist
    
    print "Smoothed prediction:\n"
    print "\t-Segmentation:\n"
    print "\t    -L1:"
    print "\t        Best threshold: %d" % T_smoothed
    print "\t        Error: %.5f" % score_smoothed
    print "\t    -Variation of Information:"
    print "\t        Best threshold: %d"% T_smoothed_VI
    print "\t        Error: %.5f\n" % VI_smoothed
    print "\t-Distance map:\n"
    print "\t    -L1 error: %.3f" % L1_err_smoothed
    print "\t    -L2 error: %.1f" % L2_err_smoothed
    print "\t    -Percentage correct: %.2f%%" % perc_smoothed
    print "\t    -VI score: %.3f\n" % VI_smoothed_dist
        


    if PLOTTING:
        
        # Plot 3 slices throughout the cube.
        
        row_length = 3
        pl.subplot(1,row_length,1)
        
        pl.title('Predicted Distance')
        pl.imshow(np.squeeze(predicted_distances[square_slice]), vmin=0, vmax=15,interpolation='nearest')
        
        pl.subplot(1,row_length,2)
        
        pl.title('Smoothed Distance')
        pl.imshow(np.squeeze(out), vmin=0, vmax=15,interpolation='nearest')
        
        pl.subplot(1,row_length,3)
        
        pl.title('Ground Truth')
        pl.imshow(np.squeeze(true_distances[square_slice]), vmin=0, vmax=15,interpolation='nearest')
        
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
