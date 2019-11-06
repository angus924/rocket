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
* NumPy;
* scikit-learn (or equivalent).

All of these should be ready to go in [Anaconda](https://www.anaconda.com/distribution/).

For `reproduce_experiments_bakeoff.py`, we also use pandas (included in Anaconda).

For `reproduce_experiments_scalability.py`, you will also need [PyTorch](https://pytorch.org/) (1.2+).

## Basic Use

The key ROCKET functions, `generate_kernels(...)` and `apply_kernels(...)`, are contained in [`rocket_functions.py`](./code/rocket_functions.py).  A worked example is provided in the [demo](./code/demo.ipynb) notebook.

**Note**: For larger datasets, you should follow the example in [`reproduce_experiments_scalability.py`](./code/reproduce_experiments_scalability.py).  (Updated documentation is forthcoming.)

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
# if not already normalised, normalise time series (both training and test data)
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

[`reproduce_experiments_additional.py`](./code/reproduce_experiments_additional.py) is intended to allow for reproduction of the experiments on the additional 2018 datasets (using the txt versions of the relevant datasets from [timeseriesclassification.com](http://www.timeseriesclassification.com)).  The main differences from `reproduce_experiments_bakeoff.py` relate to:

* normalising time series;
* handling missing values (missing values are interpolated); and
* handing variable length time series (variable length time series are rescaled or used "as is" as determined by 10-fold cross-validation).

The arguments are the same as for `reproduce_experiments_bakeoff.py`.

Again, as ROCKET is nondeterministic, results will differ between runs.  However, any single run should produce representative results.

Examples:

```bash
python reproduce_experiments_additional.py -i ./Univariate_arff/ -o ./
python reproduce_experiments_additional.py -i ./Univariate_arff/ -o ./ -n 1 -k 100
```

### Scalability

[`reproduce_experiments_scalability.py`](./code/reproduce_experiments_scalability.py) is intended to:

* allow for reproduction of the scalability experiments (in terms of dataset size); and
* serve as a template for integrating ROCKET with logistic / softmax regression and stochastic gradient descent (or, e.g., Adam) for other large datasets using PyTorch.

The required arguments are:

* `-tr` or `--training_path`, the training dataset (csv);
* `-te` or `--test_path`, the test dataset (csv);
* `-o` or `--output_path`, the directory in which to save the results;
* `-k` or `--num_kernels`, the number of kernels.

**Note**: It may be necessary to adapt the code to your dataset in terms of dataset size and structure, regularisation, etc.

Examples:

```bash
python reproduce_experiments_scalability.py -tr training_data.csv -te test_data.csv -o ./ -k 100
python reproduce_experiments_scalability.py -tr training_data.csv -te test_data.csv -o ./ -k 1_000
python reproduce_experiments_scalability.py -tr training_data.csv -te test_data.csv -o ./ -k 10_000
```

## Contributing

For the time being, this repository will be kept 'as is', to reflect the code used for the experiments in our paper.  We intend to make a development version of ROCKET available shortly.

## Acknowledgements

We thank Professor Eamonn Keogh and all the people who have contributed to the UCR time series classification archive.  Figures in our paper showing the ranking of different classifiers and variants of ROCKET were produced using code from [Ismail Fawaz et al. (2019)](https://github.com/hfawaz/cd-diagram).

<div align="center">:rocket:</div>
