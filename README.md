# Accompanying code for ACM-BCB submission titled "Distribution-based Sketching of Single-Cell Samples"
Our code has been tested using Python 3.8.5, Anndata 0.7.6 and Numpy 1.20

## Download Datasets
Please download the datasets used in the paper from the Zenodo repository-: https://zenodo.org/record/6546964

## Sample Commands for running scripts
scripts/core/sample_set_classification.py --> driver script with options to subsample/sketch from input datasets, merge sketched files together and perform classification experiments outlined in the paper.
scripts/core/model.py --> contains the Kernel Herding Algorithm, Computation of Random Fourier Features, Cluster Centroids and Frequency vectors.
scripts/core/train.py --> Leave One Out training experiment setups
scripts/plots/metric_calc.py --> Random Function Evaluation, Singular Values and Cluster frequency calculation


Run python3 sample_set_classification.py --help for description of arguments needed.



