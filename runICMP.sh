for i in `seq 0 9`;
do
	let "j=i+1"
	j=`printf %02d $j`
	./apply_to_2D_data.py /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/predictions_ALL/prediction$i.tif /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/ground_truth_distance/${j}m.tif /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/temp /Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/smoothed_ALL/smoothed$j.tif --np --sx 100 --sy 100
done
