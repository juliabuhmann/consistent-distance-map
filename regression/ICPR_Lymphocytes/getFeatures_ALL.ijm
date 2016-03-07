function action(inputfolder, filename, outputfolder) {
	print (inputfolder+filename);
	open(inputfolder+filename);
	run("Trainable Weka Segmentation");
	selectWindow("Trainable Weka Segmentation v3.1.0");
	call("trainableSegmentation.Weka_Segmentation.setFeature", "Variance=true");
	call("trainableSegmentation.Weka_Segmentation.setFeature", "Mean=true");
	call("trainableSegmentation.Weka_Segmentation.setFeature", "Minimum=true");
	call("trainableSegmentation.Weka_Segmentation.setFeature", "Maximum=true");
	call("trainableSegmentation.Weka_Segmentation.setFeature", "Median=true");
	call("trainableSegmentation.Weka_Segmentation.setFeature", "Anisotropic_diffusion=true");
	call("trainableSegmentation.Weka_Segmentation.setFeature", "Bilateral=true");
	call("trainableSegmentation.Weka_Segmentation.setFeature", "Lipschitz=true");
	call("trainableSegmentation.Weka_Segmentation.setFeature", "Kuwahara=true");
	call("trainableSegmentation.Weka_Segmentation.setFeature", "Gabor=true");
	call("trainableSegmentation.Weka_Segmentation.setFeature", "Derivatives=true");
	call("trainableSegmentation.Weka_Segmentation.setFeature", "Laplacian=true");
	call("trainableSegmentation.Weka_Segmentation.setFeature", "Structure=true");
	call("trainableSegmentation.Weka_Segmentation.setFeature", "Entropy=true");
	call("trainableSegmentation.Weka_Segmentation.setFeature", "Neighbors=true");
	
	call("trainableSegmentation.Weka_Segmentation.saveFeatureStack", outputfolder, filename+"_");

	run("Close All");
}

input = "/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/data/";
output = "/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/features_ALL/";

list = getFileList(input);
for (i = 0; i < list.length; i++) {
        action(input, list[i], output);
}