# Eliminating Biasing Signals in Lung Cancer Images for Prognosis Predictions with Deep Learning


This repository contains the necessary files to reproduce the results of paper
"Eliminating Biasing Signals Lung Cancer Images for Prognosis Predictions with Deep Learning"
by W.A.C. van Amsterdam, J.J.C. Verhoeff, P.A. de Jong, T. Leiner and M.J.C. Eijkemans; 
in Nature Digital Medicine, 2019

## Replicating the experiments

See this release for the code that generated the published results

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3522229.svg)](https://doi.org/10.5281/zenodo.3522229)

Please follow these steps to replicate the results as published.
The original python scripts are (somewhat) self-explanatory.
They do contain unused code that was useful during initial experiments, but was not used for the final publication

### Installation

The easiest way to go about this is to create a new conda environment and install all dependencies using conda and pip

```
conda create --name elimbias
conda activate elimbias
conda install python=3.7.3 tqdm numpy pandas feather-format nibabel pillow scikit-learn tensorboard future seaborn
conda install -c pytorch pytorch=1.1.0 torchvision
pip install pyro-ppl==0.3.0 pypng pylidc
```



### Pre-processing

Go to subfolder elimbias/preproces, follow steps in README there

The goal of these steps is to end up with a collection of images that are neural-network ready, and each have associated measurements (e.g. size and variance) that can be used in a structural causal model

The result is a data folder that contains the images separated in train / valid subfolders (test is optional but not default), with associated measurements in a labels.csv file

### Data simulation

This is where the statistical association between the images and the 'clinical' data are simulated, based on a structural causal model and the measurements of the images.

1. Define a structural causal model that will generate the data

   See experiments/sims/README.md for a short instruction to define a structural causal model
   See experiments/sims for an example csv file that defines a structural causal model

2. Define a setting in the settings directory with a setting.json file that together with the structural causal model defines the experiment (see the example)

3. After defining the SCM and setting, run simulate_data.py to create a dataset based on the SCM and sample images accordingly for the defined setting like so:

   ```python simulate_data.py --setting <mysetting>```

   run without the `--setting` argument to replicate the published results, using the default setting

   This will create a data folder in the setting/mysetting folder.
   Here are the images stored, coupled with the simulated ground truth data that will be used for training and validation. 

### Running the models

To replicate, run:

```python train.py```

To run on your own simulated data:

```python train.py --setting <mysetting>```

To evaluate the CNNs ability to predict the ground truth measurements, run with: 

```python train.py --setting <mysetting> --fase feature```

Result will be saved in the setting directory, with subfolders for each 'fase' (xybn: predict x, y and use bottleneck loss; feature: predict features)

[experiments/base_model/params.json](experiments/base_model/params.json) contains the hyperparameters that controls how train.py runs

### Evaluation

Run Tensorboard in this directory for visualization of the results
