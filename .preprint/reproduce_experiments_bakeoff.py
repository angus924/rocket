# Angus Dempster, Francois Petitjean, Geoff Webb

# Dempster A, Petitjean F, Webb GI (2019) ROCKET: Exceptionally fast and
# accurate time series classification using random convolutional kernels.
# arXiv:1910.13051

import argparse
import numpy as np
import pandas as pd
import time

from sklearn.linear_model import RidgeClassifierCV

from rocket_functions import generate_kernels, apply_kernels

# == notes =====================================================================

# - This script is intended to allow for reproduction of the experiments on the
#   "bake off" datasets, using the txt versions of the "bake off" datasets from
#   timeseriesclassification.com (Univariate2018_arff.zip).
# - The required arguments for this script are:
#   - -i or --input_path, the parent directory for the datasets; and
#   - -o or --output_path, to save "bakeoff_results.csv".
# - Optional arguments allow you to set the number of runs, -n or --num_runs,
#   and the number of kernels, -k or --num_kernels.
# - If input_path is ".../Univariate_arff/", then each dataset should be
#   located at "{input_path}/{dataset_name}/{dataset_name}_TRAIN.txt", etc.
# - Most of the code herein relates to handling input/output, multiple runs,
#   timings, etc.  The key code to use ROCKET is simply:
#       kernels = generate_kernels(X_training.shape[1], 10_000)
#       X_training_transform = apply_kernels(X_training, kernels)
#       classifier.fit(X_training_transform, Y_training)
#       X_test_transform = apply_kernels(X_test)
#       classifier.predict(X_test_transform)

# == parse arguments ===========================================================

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_path", required = True)
parser.add_argument("-o", "--output_path", required = True)
parser.add_argument("-n", "--num_runs", type = int, default = 10)
parser.add_argument("-k", "--num_kernels", type = int, default = 10_000)

arguments = parser.parse_args()

# == bake off dataset names ====================================================

dataset_names_bakeoff = \
(
    "Adiac",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "Car",
    "CBF",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "GunPoint",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "InlineSkate",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "Plane",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "ScreenType",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "UWaveGestureLibraryAll",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga"
)

# == bake off convenience function =============================================

def run(training_data, test_data, num_runs = 10, num_kernels = 10_000):

    results = np.zeros(num_runs)
    timings = np.zeros([4, num_runs]) # training transform, test transform, training, test

    Y_training, X_training = training_data[:, 0].astype(np.int), training_data[:, 1:]
    Y_test, X_test = test_data[:, 0].astype(np.int), test_data[:, 1:]

    for i in range(num_runs):

        input_length = X_training.shape[1]
        kernels = generate_kernels(input_length, num_kernels)

        # -- transform training ------------------------------------------------

        time_a = time.perf_counter()
        X_training_transform = apply_kernels(X_training, kernels)
        time_b = time.perf_counter()
        timings[0, i] = time_b - time_a

        # -- transform test ----------------------------------------------------

        time_a = time.perf_counter()
        X_test_transform = apply_kernels(X_test, kernels)
        time_b = time.perf_counter()
        timings[1, i] = time_b - time_a

        # -- training ----------------------------------------------------------

        time_a = time.perf_counter()
        classifier = RidgeClassifierCV(alphas = 10 ** np.linspace(-3, 3, 10), normalize = True)
        classifier.fit(X_training_transform, Y_training)
        time_b = time.perf_counter()
        timings[2, i] = time_b - time_a

        # -- test --------------------------------------------------------------

        time_a = time.perf_counter()
        results[i] = classifier.score(X_test_transform, Y_test)
        time_b = time.perf_counter()
        timings[3, i] = time_b - time_a

    return results, timings

# == run through the bake off datasets =========================================

results_bakeoff = pd.DataFrame(index = dataset_names_bakeoff,
                               columns = ["accuracy_mean",
                                          "accuracy_standard_deviation",
                                          "time_training_seconds",
                                          "time_test_seconds"],
                               data = 0)
results_bakeoff.index.name = "dataset"

compiled = False

print(f"RUNNING".center(80, "="))

for dataset_name in dataset_names_bakeoff:

    print(f"{dataset_name}".center(80, "-"))

    # -- read data -------------------------------------------------------------

    print(f"Loading data".ljust(80 - 5, "."), end = "", flush = True)

    training_data = np.loadtxt(f"{arguments.input_path}/{dataset_name}/{dataset_name}_TRAIN.txt")
    test_data = np.loadtxt(f"{arguments.input_path}/{dataset_name}/{dataset_name}_TEST.txt")

    print("Done.")

    # -- precompile ------------------------------------------------------------

    if not compiled:

        print(f"Compiling ROCKET functions (once only)".ljust(80 - 5, "."), end = "", flush = True)

        _ = generate_kernels(100, 10)
        apply_kernels(np.zeros_like(training_data)[:, 1:], _)
        compiled = True

        print("Done.")

    # -- run -------------------------------------------------------------------

    print(f"Performing runs".ljust(80 - 5, "."), end = "", flush = True)

    results, timings = run(training_data, test_data,
                           num_runs = arguments.num_runs,
                           num_kernels = arguments.num_kernels)
    timings_mean = timings.mean(1)

    print("Done.")

    # -- store results ---------------------------------------------------------

    results_bakeoff.loc[dataset_name, "accuracy_mean"] = results.mean()
    results_bakeoff.loc[dataset_name, "accuracy_standard_deviation"] = results.std()
    results_bakeoff.loc[dataset_name, "time_training_seconds"] = timings_mean[[0, 2]].sum()
    results_bakeoff.loc[dataset_name, "time_test_seconds"] = timings_mean[[1, 3]].sum()

print(f"FINISHED".center(80, "="))

results_bakeoff.to_csv(f"{arguments.output_path}/results_bakeoff.csv")
