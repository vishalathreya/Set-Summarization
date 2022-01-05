# Set Summarization using Kernel Herding & Random Fourier Features

## Sample Commands for running scripts

### Subsample
Runs subsampling using individual methods and saves the original subsampled features and their corresponding Random Fourier Features

python3 sample_set_classification.py 0 100 15 subsample 1 3 500 && python3 sample_set_classification.py 0 100 15 subsample 2 3 500 && python3 sample_set_classification.py 0 100 15 subsample 0.5 3 500 && python3 sample_set_classification.py 0 100 15 subsample 0.2 3 500 && python3 sample_set_classification.py 0 100 15 subsample 1 2 500 && python3 sample_set_classification.py 0 100 15 subsample 2 2 500 && python3 sample_set_classification.py 0 100 15 subsample 0.5 2 500 && python3 sample_set_classification.py 0 100 15 subsample 0.2 2 500 && python3 sample_set_classification.py 0 100 15 subsample 1 1 500 && python3 sample_set_classification.py 0 100 15 subsample 2 1 500 && python3 sample_set_classification.py 0 100 15 subsample 0.5 1 500 && python3 sample_set_classification.py 0 100 15 subsample 0.2 1 500

### Merge
Merges individual subsampled sets into 1 .h5ad and .npy files

python3 sample_set_classification.py 0 100 10 merge 1 3 500 && python3 sample_set_classification.py 0 100 10 merge 2 3 500 && python3 sample_set_classification.py 0 100 10 merge 0.5 3 500 && python3 sample_set_classification.py 0 100 10 merge 0.2 3 500 && python3 sample_set_classification.py 0 100 10 merge 1 2 500 && python3 sample_set_classification.py 0 100 10 merge 2 2 500 && python3 sample_set_classification.py 0 100 10 merge 0.5 2 500 && python3 sample_set_classification.py 0 100 10 merge 0.2 2 500 && python3 sample_set_classification.py 0 100 10 merge 1 1 500 && python3 sample_set_classification.py 0 100 10 merge 2 1 500 && python3 sample_set_classification.py 0 100 10 merge 0.5 1 500 && python3 sample_set_classification.py 0 100 10 merge 0.2 1 500

### Classify
python3 sample_set_classification.py 0 100 15 classify 1 1 500
