# Demos

The demos showcase the application of HEMnet in predicting cancer regions on standard histopathology images, like those 
from [The Cancer Genome Atlas](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga).

Get started instantly by clicking on a badge next to one of the demo titles below. These demos run in the browser - you do not
need to install anything locally.  


## Cancer Prediction on TCGA slide [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BiomedicalMachineLearning/HEMnet/blob/master/Demo/TCGA_Inference.ipynb)

Predict cancer regions on a [colon adenocarcinoma slide](https://portal.gdc.cancer.gov/files/1f15485a-15dd-460d-b6a3-97e999c07a68) 
from TCGA using a pretrained colorectal cancer model developed with HEMnet. 

We walk you through the code, step-by-step, starting from
 loading the software, slide and model. Next we normalise and tile the slide before predicting cancer regions. Finally, 
 we calculate the proportion of tissue area occupied by the cancer. 
 
 > It takes about 1.5hrs to run it all but you can use other browser tabs and other programs on your computer during 
> the long processing steps, which have time estimates and progress bars. Just don't close the running colab tab. 

Extend our demo to apply our model to other colorectal cancer slides from TCGA. Use the 
[GDC Data Portal](https://portal.gdc.cancer.gov/repository?facetTab=files&filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22TCGA-COAD%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_type%22%2C%22value%22%3A%5B%22Slide%20Image%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.experimental_strategy%22%2C%22value%22%3A%5B%22Diagnostic%20Slide%22%5D%7D%7D%5D%7D&searchTableTab=cases)
to find slides and add them to your cart. In your cart, download the manifest file and follow instructions in the 
colab notebook to upload and use the manifest to load your slides. 

## Tile Prediction [![launch ImJoy](https://imjoy.io/static/badge/launch-imjoy-badge.svg)](https://imjoy.io/#/app?plugin=https://github.com/BiomedicalMachineLearning/HEMnet/blob/master/Demo/HEMnet_Tile_Predictor.imjoy.html)

Predict the cancer status of a single 10x magnification tile using a pretrained colorectal cancer model developed with HEMnet. 

Use the graphical user interface in ImJoy to predict on a preloaded cancer tile or upload your own tile from your computer. 