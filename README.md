# Brain Age Prediction Residual Neural Network

## Description
A 3D residual neural network (ResNet) trained on MRI images to perform brain age prediction implemented using TensorFlow (version 2.1.0). The method was trained on four image types: raw T1 images, Jacobian maps, and gray and white matter segmentation maps. In our experiments, the ResNet was trained and evaluated on an Icelandic brain MRI dataset (1,264 healthy subjects) and the [IXI dataset](https://brain-development.org/ixi-dataset/) (440 images), and then used to generating brain age predictions for the [UK Biobank](https://www.ukbiobank.ac.uk/register-apply/) (19,642 subjects). For a detailed discription of the method see [Brain age prediction using deep learning uncovers associated sequence variants](https://www.nature.com/articles/s41467-019-13163-9).

## Notebooks
Notebooks detailing the training of the ResNets and generation of UK Biobank predictions are provided for the four image types: [raw T1 images](Code/ResNet_TrainingAndInference(RawT1).ipynb), [Jacobian maps](Code/ResNet_TrainingAndInference(Jacobian).ipynb), [gray matter segmentation maps](Code/ResNet_TrainingAndInference(GrayMatter).ipynb), [white matter segmentation maps](Code/ResNet_TrainingAndInference(WhiteMatter).ipynb).

## Pretrained Networks
Pretrained 3D residual networks can be found in the Models directory.

## Setup

### Preprocessing
All MRI images need to be preprocessed using the [CAT12 toolbox](http://www.neuro.uni-jena.de/cat/). The MRI template that was used for coregisteration has been removed in the latest version of CAT12 (Template_1_IXI555_MNI152.nii). This template can be found in [Version 12.6](https://zenodo.org/record/2573967#.Yinrs4nP0uW) of CAT12. However, due to small differences between CAT12 versions, it is likly that the pretrained networks work best with the same CAT12 version that was used during training ([Version 1092](http://www.neuro.uni-jena.de/cat12/cat12_r1092.zip)).

### Replacing the training data
The Icelandic dataset that was use for training is not publicly available, however, it can be replaced with any sufficiantly large MRI dataset. The training code provided in the notebooks can be reused by replacing the train, val, test data frames with new data. For new data frames it is necessary to proved four columns: **Loc**, **Scanner**, **Gender**, **Age**. 

* **Loc** should include all paths to the preprocessed MRI NIFTI files.
* **Scanner** is an integer (0 or 1) specifiying the MRI scanner source.
* **Gender** should be an integer specifying the subjects sex (0 if male, 1 if female).
* **Age** should specify the subjects age during the imaging visit. 

Note that the current ResNet version can only handle two scanner sources. In cases were the dataset includes more than two sources it is necessary to one hot encode the scanner variable and add more inputs to the ResNet. Alternativly, the scanner variable can be replaced with some other variable of interest, e.g., total intracranial volume.

### Generating brain age predictions for the UK Biobank
To generate brain age predictions for the UK Biobank data it is necessary to download the MRI brain scans from [UK Biobank](https://www.ukbiobank.ac.uk/register-apply/) and place the preprocessed NIFTI files in the folder Data/CAT_UK_Biobank.

## Citing

 If you find this work useful in your research, please consider citing: 

    @article{jonsson2019brain,
      title={Brain age prediction using deep learning uncovers associated sequence variants},
      author={J{\'o}nsson, Benedikt Atli and Bjornsdottir, Gyda and Thorgeirsson, TE and Ellingsen, Lotta Mar{\'\i}a and Walters, G Bragi and Gudbjartsson, DF and Stefansson, Hreinn and Stefansson, Kari and Ulfarsson, MO},
      journal={Nature communications},
      volume={10},
      number={1},
      pages={1--10},
      year={2019},
      publisher={Nature Publishing Group}
    }

## Todo
* Add code for models trained on SBM and VBM features, such as, Gaussian process regression and SVR.
