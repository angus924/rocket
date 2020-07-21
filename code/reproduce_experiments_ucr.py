# Angus Dempster, Francois Petitjean, Geoff Webb
#
# @article{dempster_etal_2020,
#   author  = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
#   title   = {ROCKET: Exceptionally fast and accurate time classification using random convolutional kernels},
#   year    = {2020},
#   journal = {Data Mining and Knowledge Discovery},
#   doi     = {https://doi.org/10.1007/s10618-020-00701-z}
# }
#
# https://arxiv.org/abs/1910.13051 (preprint)

import argparse
import numpy as np
import pandas as pd
import time

from sklearn.linear_model import RidgeClassifierCV

from rocket_functions import generate_kernels, apply_kernels

# == notes =====================================================================

# Reproduce the experiments on the UCR archive.
#
# For use with the txt version of the datasets (timeseriesclassification.com)
# and, for datasets with missing values and/or variable-length time series,
# with missing values interpolated, and variable-length time series padded to
# the same length as the longest time series per the version of the datasets as
# per https://www.cs.ucr.edu/~eamonn/time_series_data_2018/.
#
# Arguments:
# -d --dataset_names : txt file of dataset names
# -i --input_path    : parent directory for datasets
# -o --output_path   : path for results
# -n --num_runs      : number of runs (optional, default 10)
# -k --num_kernels   : number of kernels (optional, default 10,000)
#
# *dataset_names* should be a txt file of dataset names, each on a new line.
#
# If *input_path* is, e.g., ".../Univariate_arff/", then each dataset should be
# located at "{input_path}/{dataset_name}/{dataset_name}_TRAIN.txt", etc.

# == parse arguments ===========================================================

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset_names", required = True)
parser.add_argument("-i", "--input_path", required = True)
parser.add_argument("-o", "--output_path", required = True)
parser.add_argument("-n", "--num_runs", type = int, default = 10)
parser.add_argument("-k", "--num_kernels", type = int, default = 10_000)

arguments = parser.parse_args()

# == run =======================================================================

dataset_names = np.loadtxt(arguments.dataset_names, "str")

results = pd.DataFrame(index = dataset_names,
                       columns = ["accuracy_mean",
                                  "accuracy_standard_deviation",
                                  "time_training_seconds",
                                  "time_test_seconds"],
                       data = 0)
results.index.name = "dataset"

print(f"RUNNING".center(80, "="))

for dataset_name in dataset_names:

    print(f"{dataset_name}".center(80, "-"))

    # -- read data -------------------------------------------------------------

    print(f"Loading data".ljust(80 - 5, "."), end = "", flush = True)

    training_data = np.loadtxt(f"{arguments.input_path}/{dataset_name}/{dataset_name}_TRAIN.txt")
    Y_training, X_training = training_data[:, 0].astype(np.int32), training_data[:, 1:]

    test_data = np.loadtxt(f"{arguments.input_path}/{dataset_name}/{dataset_name}_TEST.txt")
    Y_test, X_test = test_data[:, 0].astype(np.int32), test_data[:, 1:]

    print("Done.")

    # -- run -------------------------------------------------------------------

    print(f"Performing runs".ljust(80 - 5, "."), end = "", flush = True)

    _results = np.zeros(arguments.num_runs)
    _timings = np.zeros([4, arguments.num_runs]) # trans. tr., trans. te., training, test

    for i in range(arguments.num_runs):

        input_length = X_training.shape[-1]
        kernels = generate_kernels(input_length, arguments.num_kernels)

        # -- transform training ------------------------------------------------

        time_a = time.perf_counter()
        X_training_transform = apply_kernels(X_training, kernels)
        time_b = time.perf_counter()
        _timings[0, i] = time_b - time_a

        # -- transform test ----------------------------------------------------

        time_a = time.perf_counter()
        X_test_transform = apply_kernels(X_test, kernels)
        time_b = time.perf_counter()
        _timings[1, i] = time_b - time_a

        # -- training ----------------------------------------------------------

        time_a = time.perf_counter()
        classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
        classifier.fit(X_training_transform, Y_training)
        time_b = time.perf_counter()
        _timings[2, i] = time_b - time_a

        # -- test --------------------------------------------------------------

        time_a = time.perf_counter()
        _results[i] = classifier.score(X_test_transform, Y_test)
        time_b = time.perf_counter()
        _timings[3, i] = time_b - time_a

    print("Done.")

    # -- store results ---------------------------------------------------------

    results.loc[dataset_name, "accuracy_mean"] = _results.mean()
    results.loc[dataset_name, "accuracy_standard_deviation"] = _results.std()
    results.loc[dataset_name, "time_training_seconds"] = _timings.mean(1)[[0, 2]].sum()
    results.loc[dataset_name, "time_test_seconds"] = _timings.mean(1)[[1, 3]].sum()

print(f"FINISHED".center(80, "="))

results.to_csv(f"{arguments.output_path}/results_ucr.csv")
