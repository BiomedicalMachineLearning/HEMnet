[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BiomedicalMachineLearning/HEMnet/master?filepath=Development)
[![launch ImJoy](https://imjoy.io/static/badge/launch-imjoy-badge.svg)](https://imjoy.io/#/app?plugin=https://github.com/BiomedicalMachineLearning/HEMnet/blob/master/Demo/HEMnet_Tile_Predictor.imjoy.html)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BiomedicalMachineLearning/HEMnet/blob/master/Demo/TCGA_Inference.ipynb)

# HEMnet - Haematoxylin & Eosin and Molecular neural network

## Description
A deep learning automated cancer diagnosis software using molecular labelling to improve pathological annotation of 
Haematoxylin and Eosin (H&E) stained tissue. 

## Installation

1. Docker

    You can download and run the docker image using the following commands:
    
    ```
    docker pull andrewsu1/hemnet    
    docker run -it andrewsu1/hemnet
    ```
2. Conda 
   
   Install Openslide (this is necessary to open whole slide images) - download it [here](https://openslide.org/download/)
   
   Create a conda environment from the `environment.yml` file
   
   ```
   conda env create -f environment.yml
   conda activate HEMnet
   ```
   
## Usage
### Slide Preparation

Name slides in the format: `slide_id_TP53` for TP53 slides and `slide_id_HandE` for H&E slides
The `TP53` and `HandE` suffix is used by HEMnet to identify the stain used. 

### 1. Generate training dataset

`python HEMnet_train_dataset.py -b /path/to/base/directory -s relative/path/to/slides -o relative/path/to/output/directory 
-t relative/path/to/template_slide.svs -v`

### 2. Generate test dataset
`python HEMnet_test_dataset.py -b /path/to/base/directory -s /relative/path/to/slides -o /relative/path/to/output/directory
 -t relative/path/to/template_slide -m tile_mag -a align_mag -c cancer_thresh -n non_cancer_thresh`
 * `-t` is the relative path to the template slide from which all other slides will be normalised against. 
 This should be same as the template slide used for generating the train dataset. 
 * `-m` is the tile magnification. e.g. if  the input is `10` then the tiles will be output at 10x
 * `-a` is the align magnification. Paired TP53 and H&E slides will be registered at this magnification. 
 To reduce computation time we recommend this be less than the tile magnification - a five times downscale generally works well.
 * `-c` cancer threshold to apply to the DAB channel. DAB intensities less than this threshold indicate cancer.
 * `-n` non-cancer threshold to apply to the DAB channel. DAB intensities greater than this threshold indicate no cancer. 
### 3. Train model
 
### 4. Apply model to diagnose new images

## Results

## Citing HEMnet

## The Team
Please contact Dr Quan Nguyen (quan.nguyen@uq.edu.au), Andrew Su (a.su@uqconnect.edu.au), 
and Xiao Tan (xiao.tan@uqconnect.edu.au) for issues, suggestions, 
and we are very welcome to collaboration opportunities.

