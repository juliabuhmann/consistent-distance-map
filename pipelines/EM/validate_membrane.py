__author__ = 'julia'

import os
import experiments_organization as eo
import h5py
import surface_reconstruction as sr
from scipy import ndimage
import numpy as np
from gala import evaluate
import matplotlib.pyplot as plt
import boundary_fscore

AFFINITYGRAPH = False
PLOT = True
MEMBRANEVALIDATION = True
OPTIMIZETHRESHOLD = True

length = [100, 100, 50]
# length = [10, 10, 5]
pos = [70, 30, 50]
pos = [40, 40, 10]

bin_classfilename = '/raid/julia/projects/LSTM/membrane_distance/' \
                    '20160307/inference_results/093941/feature_collection_testing.h5'

f = h5py.File(bin_classfilename, 'r')
aff = f['inference_results'].value
f.close()

# bin_classfilename = '/raid/julia/projects/LSTM/data/' \
#                     'affinity_graph_inference_google/affinity_graphs_mean/cube26.h5'
# f = h5py.File(bin_classfilename, 'r')
# aff = f['meanxyz'].value
# f.close()

print aff.shape

inputfile = '/raid/julia/projects/LSTM/surface_reconstruction/20160307/194212/result.h5'
f = h5py.File(inputfile, 'r')
true_distance = f['gt'].value
predicted_distances = f['noisy_distance'].value
out = f['reconstructed'].value
f.close()

# Load Ground Truth
inputfilename = '/raid/julia/projects/LSTM/data/ground_truth_old_numbering/cube26_neuron.h5'
f = h5py.File(inputfilename, 'r')
seg_gt = f['seg'].value
f.close()

cube_slice = [slice(pos[0],pos[0]+length[0]),
              slice(pos[1],pos[1]+length[1]),
              slice(pos[2],pos[2]+length[2])]


seg_gt = seg_gt[cube_slice]
aff = aff[cube_slice]

# Convert probmaps into segmentations
def from_prob_map_to_seg(prob_map, threshold=0.4, verbose=True):
    cc_out, count = ndimage.label(np.invert(prob_map<threshold))
    if verbose:
        print "%i different labels" %count
    return cc_out


def from_probmap_to_boundary(probmap, lower_threshold=0.0, upper_threshold=0.9):
    boundary_seg = probmap.copy()
    boundary_seg[np.where(probmap >upper_threshold)] = 0
    boundary_seg[np.where(probmap < lower_threshold)] = 0
    boundary_seg = boundary_seg > 0
    return boundary_seg*1

def from_segmentation_to_boundary(seg, iterations=1):
    # Get the boundary from a volumetric
    # object matrix (each label corresponds to one object)

    new_boundary_matrix = np.zeros_like(seg)
    for object_number in np.unique(seg)[1:]:
#         object_number = 10
        single_object = seg.copy()
        single_object[np.where(seg != object_number)] = 1
        single_object[np.where(seg == object_number)] = 0
        single_object_eroded = ndimage.morphology.binary_dilation(single_object,
                                                           iterations=iterations)

        new_boundary =  single_object*1 - single_object_eroded*1
        new_boundary_matrix += new_boundary.astype(seg.dtype)
    new_boundary_matrix = (new_boundary_matrix > 0)*1
    print np.unique(new_boundary_matrix)
    return new_boundary_matrix*1

# seg_aff = from_prob_map_to_seg(aff, 0.4)
# seg_rec = from_prob_map_to_seg(out, 5)
gt_boundary = from_segmentation_to_boundary(seg_gt)
bin_boundary = from_probmap_to_boundary(aff,
                                        lower_threshold=0.0, upper_threshold=0.9)
rec_boundary = from_probmap_to_boundary(out, lower_threshold=0, upper_threshold=1)
noisy_boundary = from_probmap_to_boundary(predicted_distances,
                                          lower_threshold=0, upper_threshold=4)

print np.unique(rec_boundary)
print np.unique(gt_boundary)
print np.unique(bin_boundary)

def optimize_threshold(prob_map, gt_boundary, steps= 10):
    fscores = []
    thresholds = []
    precisions = []
    recalls = []
    if np.amax(prob_map) > 1:
        step_size = 1
    else:
        step_size = np.amax(prob_map) /float(steps)
    print step_size
    for low_th in np.arange(0, np.amax(prob_map)+step_size, step_size):
        for up_th in np.arange(0, np.amax(prob_map)+step_size, step_size):
            if low_th < up_th:
                # low_th = 0.5
                # up_th  = 0.9
                boundary_res= from_probmap_to_boundary(prob_map,
                                        lower_threshold=low_th, upper_threshold=up_th)
                fscore_bin, precision, recall = boundary_fscore.fscore(boundary_res, gt_boundary, tolerance=tolerance_value)
                fscores.append(fscore_bin[2])
                precisions.append(precision)
                recalls.append(recall)
                thresholds.append((low_th, up_th))

    ind = np.argmax(fscores)
    print 'best thresholds', thresholds[ind]
    print 'best fscore', fscores[ind]
    print 'best precision', precisions[ind]
    print 'best recall', recalls[ind]
    return thresholds[ind], fscores[ind]


tolerance_value = 0
if OPTIMIZETHRESHOLD:
    print 'binary'
    thresholds_bin, fscore_bin = optimize_threshold(aff, gt_boundary)
    print 'reconstructed'
    thresholds_rec, fscore_rec = optimize_threshold(out, gt_boundary)
    print 'noisy'
    thresholds_noisy, fscore_noisy = optimize_threshold(predicted_distances, gt_boundary)

    rec_boundary = from_probmap_to_boundary(out,
                                            lower_threshold=thresholds_rec[0],
                                            upper_threshold=thresholds_rec[1])

    noisy_boundary = from_probmap_to_boundary(predicted_distances,
                                          lower_threshold=thresholds_noisy[0],
                                          upper_threshold=thresholds_noisy[1])

    # thresholds_bin = (0.5, 0.9)
    bin_boundary = from_probmap_to_boundary(aff,
                                          lower_threshold=thresholds_bin[0],
                                          upper_threshold=thresholds_bin[1])
if PLOT:
    section_slice = 10
    fig, axes = plt.subplots(ncols=4, figsize=(25, 10))
    ax1, ax2, ax3, ax4 = axes

    ax1.matshow(gt_boundary[:, :, section_slice],  cmap=plt.cm.jet)
    ax1.set_title('ground truth')
    plt.axis('off')

    ax2.matshow(bin_boundary[:, :, section_slice])
    ax2.set_title('binary fscore: %0.2f' %(fscore_bin))

    ax3.matshow(noisy_boundary[:, :, section_slice], label='test')
    ax3.set_title('noisy fscore: %0.2f' %(fscore_noisy))

    ax4.matshow(rec_boundary[:, :, section_slice],  cmap=plt.cm.jet)
    ax4.set_title('reconstructed  fscore: %0.2f' %(fscore_rec))
    plt.axis('off')
    plt.show()


if AFFINITYGRAPH:




    # CC_VI_pred, T_pred_CC_VI = sr.best_thresh(out, seg[cube_slice], score_func='CC_VI')
    CC_VI_pred = sr.score(out<4, seg[cube_slice]<1,  score='CC_VI')
    cc_out, count = ndimage.label(np.invert(out<5))
    vi1 = evaluate.split_vi(cc_out, seg[cube_slice])
    plt.matshow(cc_out[:, :, 0])
    plt.show()

    plt.matshow(seg[cube_slice][:, :, 0])
    plt.show()

    plt.matshow(out[:, :, 0])
    plt.show()
    print vi1
    print "\t SMOOTHED DISTANCE MAP"
    print "\t    -Variation of Information on Connected Components:"
    # print "\t        Best threshold: %d"% T_pred_CC_VI
    print "\t        Error: %.5f\n" % CC_VI_pred
    print "\t        Error VI Gala impl false merger: %.5f, false splits: %5f\n" %(vi1[0], vi1[1])




    # CC_VI_pred, T_pred_CC_VI = sr.best_thresh(aff[cube_slice], seg[cube_slice], score_func='CC_VI')
    cc_aff = aff[cube_slice] < 0.9
    cc_aff, count = ndimage.label(np.invert(cc_aff))
    CC_VI_pred = sr.score(aff[cube_slice] < 0.4, seg[cube_slice] < 1,  score='CC_VI')

    vi1 = evaluate.split_vi(cc_aff, seg[cube_slice])
    plt.matshow(cc_aff[:, :, 0])
    plt.show()


    print "\t BINARY OUTPUTMAP"
    print "\t    -Variation of Information on Connected Components:"
    # print "\t        Best threshold: %.5f"% T_pred_CC_VI
    print "\t        Error: %.5f\n" % CC_VI_pred
    print "\t        Error VI Gala impl false merger: %.5f, false splits: %5f\n" %(vi1[0], vi1[1])