# ROCKET [+ MINIROCKET](https://github.com/angus924/minirocket)

***ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels.***

[Data Mining and Knowledge Discovery](https://link.springer.com/article/10.1007/s10618-020-00701-z) / [arXiv:1910.13051](https://arxiv.org/abs/1910.13051) (preprint)

> <div align="justify">Most methods for time series classification that attain state-of-the-art accuracy have high computational complexity, requiring significant training time even for smaller datasets, and are intractable for larger datasets.  Additionally, many existing methods focus on a single type of feature such as shape or frequency.  Building on the recent success of convolutional neural networks for time series classification, we show that simple linear classifiers using random convolutional kernels achieve state-of-the-art accuracy with a fraction of the computational expense of existing methods.  Using this method, it is possible to train and test a classifier on all 85 ‘bake off’ datasets in the UCR archive in < 2 h, and it is possible to train a classifier on a large dataset of more than one million time series in approximately 1 h.</div>

Please cite as:

```bibtex
@article{dempster_etal_2020,
  author = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
  title = {ROCKET: Exceptionally fast and accurate time classification using random convolutional kernels},
  year = {2020},
  journal = {Data Mining and Knowledge Discovery},
  doi = {https://doi.org/10.1007/s10618-020-00701-z}
}
```

## `sktime`

An implementation of ROCKET (with basic multivariate capability) is available through [sktime](https://github.com/alan-turing-institute/sktime).  See the [examples](https://github.com/alan-turing-institute/sktime/blob/master/examples/rocket.ipynb).

## [MINIROCKET](https://github.com/angus924/minirocket) \*NEW\*

[MINIROCKET](https://github.com/angus924/minirocket) is up to 75&times; faster than ROCKET on larger datasets.

## Results

### UCR Archive

* [Training / Test Split](results/results_ucr.csv)
* [Resamples](results/results_ucr_resamples.csv)

### Scalability

* [Training Set Size](results/results_scalability_training_set_size.csv)
* [Time Series Length](results/results_scalability_time_series_length.csv)

## Code

### [`rocket_functions.py`](code/rocket_functions.py)

### Requirements

* Python;
* Numba;
* NumPy;
* scikit-learn (or equivalent).

### Example

```python
from rocket_functions import generate_kernels, apply_kernels
from sklearn.linear_model import RidgeClassifierCV

[...] # load data, etc.

# generate random kernels
kernels = generate_kernels(X_training.shape[-1], 10_000)

# transform training set and train classifier
X_training_transform = apply_kernels(X_training, kernels)
classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
classifier.fit(X_training_transform, Y_training)

# transform test set and predict
X_test_transform = apply_kernels(X_test, kernels)
predictions = classifier.predict(X_test_transform)
```

## Reproducing the Experiments

### [`reproduce_experiments_ucr.py`](code/reproduce_experiments_ucr.py)

```
Arguments:
-d --dataset_names : txt file of dataset names
-i --input_path    : parent directory for datasets
-o --output_path   : path for results
-n --num_runs      : number of runs (optional, default 10)
-k --num_kernels   : number of kernels (optional, default 10,000)

Examples:
> python reproduce_experiments_ucr.py -d bakeoff.txt -i ./Univariate_arff -o ./
> python reproduce_experiments_ucr.py -d additional.txt -i ./Univariate_arff -o ./ -n 1 -k 1000
```

### [`reproduce_experiments_scalability.py`](code/reproduce_experiments_scalability.py)

```
Arguments:
-tr --training_path : training dataset (csv)
-te --test_path     : test dataset (csv)
-o  --output_path   : path for results
-k  --num_kernels   : number of kernels

Examples:
> python reproduce_experiments_scalability.py -tr training.csv -te test.csv -o ./ -k 100
> python reproduce_experiments_scalability.py -tr training.csv -te test.csv -o ./ -k 1000
```

## Acknowledgements

We thank Professor Eamonn Keogh and all the people who have contributed to the UCR time series classification archive.  Figures in our paper showing the ranking of different classifiers and variants of ROCKET were produced using code from [Ismail Fawaz et al. (2019)](https://github.com/hfawaz/cd-diagram).

<div align="center">:rocket:</div>
