for i in `seq 0 9`;
do
	let "j=i+1"
	j=`printf %02d $j`
	./apply_to_2D_data.py /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/predictions_rfR_noSparse_200trees/prediction$i.tif /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/ground_truth_distance/${j}m.tif /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/temp /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/smoothed_rfR_noSparse_200trees/smoothed$j.tif --np --sx 100 --sy 100
	#./apply_to_2D_data.py /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/predictions_rfR_ALL/prediction$i.tif /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/ground_truth_distance/${j}m.tif /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/temp /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/smoothed_rfR_ALL/smoothed$j.tif --np --sx 100 --sy 100
	#./apply_to_2D_data.py /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/predictions_ALL_more01_lossLAD/prediction$i.tif /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/ground_truth_distance/${j}m.tif /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/temp /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/smoothed_ALL_more01_lossLAD/smoothed$j.tif --np --sx 100 --sy 100
	#./apply_to_2D_data.py /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/predictions_ALL/prediction$i.tif /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/ground_truth_distance/${j}m.tif /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/temp /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/smoothed_ALL/smoothed$j.tif --np --sx 100 --sy 100
done
