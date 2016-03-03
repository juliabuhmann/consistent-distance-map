import numpy as np
import pylab as pl
import h5py
import os
import sys
import argparse


# -----------------------------------------------------------------------------
# 0. Define arguments that can be provided.
# -----------------------------------------------------------------------------
    
    
parser = argparse.ArgumentParser(description='Constrain distance map gradients by using Graph-Cut.')
parser.add_argument('input', metavar='path_to_input', type=str,
                   help='The path to the hdf5 file that contains the data.')
parser.add_argument('temporary_folder', metavar='path_to_a_tmp_folder', type=str, help='Path where to write some temporary data.')
parser.add_argument('-s','--size', metavar='edge_size', type=int, help='Edge size of the cube to analyze. Defaults to 24.')
parser.add_argument('--sx', '--sizex', metavar='size_x', type=int, help="Edge size of the cube to analyze along X. Defaults to 'size' if provided, or 24 if not.")
parser.add_argument('--sy', '--sizey', metavar='size_y', type=int, help="Edge size of the cube to analyze along Y. Defaults to 'size' if provided, or 24 if not.")
parser.add_argument('--sz', '--sizez', metavar='size_z', type=int, help="Edge size of the cube to analyze along Z. Defaults to 'size' if provided, or 24 if not.")
parser.add_argument('--px', '--posx', metavar='pos_x', type=int, help="Position where to start cropping cube along X. Picked at random if not provided.")
parser.add_argument('--py', '--posy', metavar='pos_y', type=int, help="Position where to start cropping cube along Y. Picked at random if not provided.")
parser.add_argument('--pz', '--posz', metavar='pos_z', type=int, help="Position where to start cropping cube along Z. Picked at random if not provided.")
parser.add_argument('-v','--verbose', action='store_true', help='Print graph cut procedure details.')
parser.add_argument('-n','--np','--noplot','--no_plot', action='store_false', help='Do not show plot.')
parser.add_argument('--ns','--noscores','--no_scores', action='store_false', help='Do not print scores.')






ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


sys.path.append(os.path.join(ROOT_PATH,'src','python'))

import validate_distance_maps as vdm
import surface_reconstruction as sr
reload(sr)

if __name__ == "__main__":
    
    # -------------------------------------------------------------------------
    # 0. Parse parameters
    # -------------------------------------------------------------------------
    args = parser.parse_args()
    
    inputfilename = args.input
    tmp_files = args.temporary_folder
    
    cube_size = args.size if args.size != None else cube_size
    cube_sizeX = args.sx if args.sx != None else cube_size
    cube_sizeY = args.sy if args.sy != None else cube_size
    cube_sizeZ = args.sz if args.sz != None else cube_size
    
    posX = args.px
    posY = args.py
    posZ = args.pz
    
    VERBOSE = args.verbose
    PLOTTING = args.np
    PRINTING = args.ns
    
    
    
    tmp_file1 = os.path.join(tmp_files,'tmp.txt')
    tmp_file2 = os.path.join(tmp_files,'tmp_output.txt')
    prog = os.path.join(ROOT_PATH,'src','cpp','graph_cut')
    
    
    
    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    
    f = h5py.File(inputfilename, 'r')
    
    predicted_distances =  f['inference_results'].value
    true_distances = f['labels'].value
    
    f.close()
    
    sx,sy,sz = predicted_distances.shape
    
    max_dist = 15
    
    start = 20
    np.random.seed(555)
    # Crop a cube of given edge.
    px = np.random.randint(0,np.maximum(sx-cube_sizeX,1)) if posX == None else posX
    py = np.random.randint(0,np.maximum(sy-cube_sizeY,1)) if posY == None else posY
    pz = np.random.randint(0,np.maximum(sz-cube_sizeZ,1)) if posZ == None else posZ
    
    pos = [px,py,pz]
    
    
    cube_slice = [slice(pos[0],pos[0]+cube_sizeX), slice(pos[1],pos[1]+cube_sizeY),slice(pos[2],pos[2]+cube_sizeZ)]
    
    out = sr.reconstruct_surface(predicted_distances[cube_slice], tmp_file1,tmp_file2,prog,overwrite=True, max_dist=max_dist, sampling = [1,1,2], verbose=VERBOSE)


    if PRINTING:
        sr.print_scores(true_distances[cube_slice], predicted_distances[cube_slice], out)
    

        

        


    if PLOTTING:
        
        # Plot 3 slices throughout the cube.
        
        row_length = 3
        sl = [int(s) for s in np.linspace(0,cube_sizeZ,5)[1:-1]]
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
        pl.suptitle(str(cube_sizeX) + ' x ' + str(cube_sizeY) + ' slices comparison')
            
            
        
        
        
        
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
        
                
        
