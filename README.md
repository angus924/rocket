# ROCKET

## Paper

This is the provisional companion repository for **Dempster A, Petitjean F, Webb GI (2019) ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels, [arXiv:1910.13051](https://arxiv.org/abs/1910.13051)**.

> <div align="justify">Most methods for time series classification that attain state-of-the-art accuracy have high computational complexity, requiring significant training time even for smaller datasets, and are intractable for larger datasets.  Additionally, many existing methods focus on a single type of feature such as shape or frequency.  Building on the recent success of convolutional neural networks for time series classification, we show that simple linear classifiers using random convolutional kernels achieve state-of-the-art accuracy with a fraction of the computational expense of existing methods.</div>

Please cite as:

```bibtex
@article{dempster_etal_2019,
  author  = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
  title   = {ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels}
  year    = {2019},
  journal = {arXiv:1910.13051}
}
```

## Results

### UCR Archive

* ['Bake off' datasets.](./results/results_ucr_bakeoff.csv)
* [Additional 2018 datasets.](./results/results_ucr_additional.csv)

### Scalability

* [Number of training examples.](./results/results_scalability_num_examples.csv)
* [Time series length.](./results/results_scalability_time_series_length.csv)

## Requirements

To use ROCKET, you will need:

* Python (3.7+);
* Numba (0.45.1+);
* NumPy; and
* scikit-learn (or equivalent).

All of these should be ready to go in [Anaconda](https://www.anaconda.com/distribution/).

## Basic Use

The key ROCKET functions, `generate_kernels(...)` and `apply_kernels(...)`, are contained in [`rocket_functions.py`](./code/rocket_functions.py).  A worked example is provided in the [demo](./code/demo.ipynb) notebook.

Basic use follows this pattern:

```python
# (1) generate random kernels
kernels = generate_kernels(input_length = X_training.shape[1], num_kernels = 10_000)

# (2) transform the training data and train a classifier
X_training_transform = apply_kernels(X = X_training, kernels = kernels)
classifier.fit(X_training_transform, Y_training)

# (3) transform the test data and use the classifier
X_test_transform = apply_kernels(X = X_test, kernels = kernels)
classifier.predict(X_test_transform)
```

**Note**: Unless already normalised (time series may already be normalised for some datasets), time series should be normalised to have a zero mean and unit standard deviation before using `apply_kernels(...)`.  For example:

```python
# if not already normalised, normalise time series (for both training and test)
X_training = (X_training - X_training.mean(axis = 1, keepdims = True)) / X_training.std(axis = 1, keepdims = True)
```

## Reproducing the Experiments

### UCR Archive

#### 'Bake Off' Datasets

[`reproduce_experiments_bakeoff.py`](./code/reproduce_experiments_bakeoff.py) is intended to allow for reproduction of the experiments on the 'bake off' datasets (using the txt versions of the 'bake off' datasets from [timeseriesclassification.com](http://www.timeseriesclassification.com)).

The required arguments are:

* `-i` or `--input_path`, the parent directory for the datasets (probably something like`.../Univariate_arff/`); and
* `-o` or `--output_path`, the directory in which to save the results.

The optional arguments are:

* `-n` or `--num_runs`, the number of runs (default 10); and
* `-k` or `--num_kernels`, the number of kernels (default 10,000).

As ROCKET is nondeterministic, results will differ between runs.  However, any single run should produce representative results.

Examples:

```bash
python reproduce_experiments_bakeoff.py -i ./Univariate_arff/ -o ./
python reproduce_experiments_bakeoff.py -i ./Univariate_arff/ -o ./ -n 1 -k 100
```

#### Additional Datasets

*(Forthcoming...)*

### Scalability

*(Forthcoming...)*

## Contributing

For the time being, this repository will be kept 'as is', to reflect the code used for the experiments in our paper.  We intend to make a development version of ROCKET available shortly.

## Acknowledgements

We thank Professor Eamonn Keogh and all the people who have contributed to the UCR time series classification archive.  Figures in our paper showing the ranking of different classifiers and variants of ROCKET were produced using code from [Ismail Fawaz et al. (2019)](https://github.com/hfawaz/cd-diagram).

<div align="center">:rocket:</div>
