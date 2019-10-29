# Preprocessing

The goal of these steps is to end up with a collection of images that are neural-network ready, and each have associated measurements (e.g. size and variance) that can be used in a structural causal model

1. Download the data from the official repository http://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX

   Note that in order for the package pylidc to work, it needs to know where the original data are stored; See https://pylidc.github.io/install.html for instructions

2. Run lidc-preprocessing.py

   This step extracts individual nodules from the ct scans and generates the 2D images from the 3D nodules. On my machine (12 threads) this takes about (nodules = 5/10 mins)

3. measure_slices.py

   Measure size (area) and variance of the pixel intensities based on the segmentations.
   These measurements will form the basis of the simulations

4. preprare-data-2d.py

   Split the data in train/valid, move to new folder, filter out slices that are too small (<20mm) or that the annotators dont agree on and normalize measurements to approximately normal distributions