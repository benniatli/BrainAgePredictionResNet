## Description
Code from the article Brain age prediction using deep learning uncovers associated sequence variants (https://www.nature.com/articles/s41467-019-13163-9). Additionally, we provide pretrained 3D residual networks (ResNets) that were trained on an Icelandic brian MRI sample of 1264 healthy subjects and the IXI dataset (https://brain-development.org/ixi-dataset/). These ResNets were trained individually on raw T1 images, Jacobian maps, and gray and white matter segmentation maps.

To generate brain age predictions for the UK Biobank sample it is necessary to download the MRI brain scans from UKB (https://www.ukbiobank.ac.uk/register-apply/) and preprocess them using the CAT12 toolbox (http://www.neuro.uni-jena.de/cat/).

#### Todo:
* Add ResNet brain age prediction training example.