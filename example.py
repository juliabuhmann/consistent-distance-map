import numpy as np
import pylab as pl
import h5py
import os
import sys

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


sys.path.append(os.path.join(ROOT_PATH,'src','python'))

import validate_distance_maps as vdm
import surface_reconstruction as sr


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


    
    
    

        
    score_pred, T_pred = sr.best_thresh(predicted_distances[cube_slice], true_distances[cube_slice],max_dist, score_func='L1')
    VI_pred, T_pred_VI = sr.best_thresh(predicted_distances[cube_slice], true_distances[cube_slice],max_dist, score_func='VI')
    
    VI_pred_dist = sr.score(predicted_distances[cube_slice], true_distances[cube_slice],'VI')
    L1_err_pred = sr.score(predicted_distances[cube_slice], true_distances[cube_slice],'L1')
    L2_err_pred = sr.score(predicted_distances[cube_slice], true_distances[cube_slice],'L2')
    perc_pred = vdm.calculate_perc_of_correct(true_distances[cube_slice].astype(np.int), predicted_distances[cube_slice].astype(np.int))*100
    
    assert L1_err_pred == vdm.calculate_L1(true_distances[cube_slice], predicted_distances[cube_slice])
    assert L2_err_pred == vdm.calculate_L2(true_distances[cube_slice], predicted_distances[cube_slice])
    
    
    score_smoothed, T_smoothed = sr.best_thresh(out, true_distances[cube_slice],max_dist, score_func='L1')
    VI_smoothed, T_smoothed_VI = sr.best_thresh(out, true_distances[cube_slice],max_dist, score_func='VI')
    
    VI_smoothed_dist = sr.score(out, true_distances[cube_slice],'VI')
    L1_err_smoothed = sr.score(out, true_distances[cube_slice],'L1')
    L2_err_smoothed = sr.score(out, true_distances[cube_slice],'L2')
    perc_smoothed = vdm.calculate_perc_of_correct(true_distances[cube_slice].astype(np.int), out.astype(np.int))*100
    
    assert L1_err_smoothed == vdm.calculate_L1(true_distances[cube_slice], out)
    assert L2_err_smoothed == vdm.calculate_L2(true_distances[cube_slice], out)
    print L1_err_smoothed,'==', vdm.calculate_L1(true_distances[cube_slice], out)
    
    
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
        sl = [int(s) for s in np.linspace(0,cube_size,5)[1:-1]]
        for i in range(3):
            
            pl.subplot(3,row_length,i*row_length+1)
            
            if i == 0:
                pl.title('Predicted Distance')
            if len(np.shape(out)) == 3:
                slices_cube = cube_slice[:-1] + [pos[2]+sl[i]]
            else:
                slices_cube = [None,None]
            pl.imshow(np.transpose(np.squeeze(predicted_distances[slices_cube])), vmin=0, vmax=15,interpolation='nearest')
            
            pl.subplot(3,row_length,i*row_length+2)
            
            if i == 0:
                pl.title('Smoothed Distance')
            if len(np.shape(out)) == 3:
                slices = [slice(None),slice(None),sl[i]]
            else:
                slices = [None,None]
            pl.imshow(np.transpose(np.squeeze(out[slices])), vmin=0, vmax=15,interpolation='nearest')
            
            pl.subplot(3,row_length,i*row_length+3)
            
            if i == 0:
                pl.title('Ground Truth')
            if len(np.shape(out)) == 3:
                slices_cube = cube_slice[:-1] + [pos[2]+sl[i]]
            else:
                slices_cube = [None,None]
            pl.imshow(np.transpose(np.squeeze(true_distances[slices_cube])), vmin=0, vmax=15,interpolation='nearest')
            max_dist+1, max_dist+1
        
        
        
        
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
        
                
        
