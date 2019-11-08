# Angus Dempster, Francois Petitjean, Geoff Webb

# Dempster A, Petitjean F, Webb GI (2019) ROCKET: Exceptionally fast and
# accurate time series classification using random convolutional kernels.
# arXiv:1910.13051

#A multichannel version of Rocket

from numba import njit, prange
import numpy as np

def generate_kernels(input_shape, num_kernels, dtype=np.float32):
    if type(input_shape)==int:
        channels, input_length = 1, input_shape
    elif type(input_shape)==tuple:
        channels, input_length = input_shape
    else:
        return
    candidate_lengths = np.array((7, 9, 11))

    # initialise kernel parameters
    weights = np.zeros((num_kernels, channels, candidate_lengths.max()), dtype = dtype) # see note
    lengths = np.zeros(num_kernels, dtype = np.int32) # see note
    biases = np.zeros(num_kernels, dtype = dtype)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)

    # note: only the first *lengths[i]* values of *weights[i]* are used

    for i in range(num_kernels):

        length = np.random.choice(candidate_lengths)
        _weights = np.random.normal(0, 1, (channels, length))
        bias = np.random.uniform(-1, 1, 1)
        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) // (length - 1)))
        padding = ((length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        weights[i,:,:length] = _weights - _weights.mean(axis=1, keepdims=True)
        lengths[i], biases[i], dilations[i], paddings[i] = length, bias, dilation, padding

    return weights, lengths, biases, dilations, paddings

@njit(fastmath = True)
def apply_kernel(X, weights, length, bias, dilation, padding):
    # print(f'X: {X.shape}')
    # print(f'w: {weights.shape}')
    channels, input_length = X.shape
    # zero padding
    if padding > 0:
        _X = np.zeros((channels, input_length + (2 * padding)), dtype=np.float32)
        _X[:,padding:(padding + input_length)] = X
        X = _X

    _, input_length = X.shape

    output_length = input_length - ((length - 1) * dilation)
    _ppv = 0 # "proportion of positive values"
    _max = np.NINF

    for i in range(output_length):

        _sum = bias

        for j in range(length):
            for c in range(channels):
                _sum += weights[c, j] * X[c, i + (j * dilation)]

        if _sum > 0:
            _ppv += 1

        if _sum > _max:
            _max = _sum

    return _ppv / output_length, _max

def apply_kernels(X, kernels):
    if len(X.shape)<3:
        return _apply_kernels(X[:, None, :].astype(np.float32), kernels)
    else:
        return _apply_kernels(X.astype(np.float32), kernels)

@njit(parallel = True, fastmath = True)
def _apply_kernels(X, kernels):
    weights, lengths, biases, dilations, paddings = kernels

    num_examples = len(X)
    num_kernels = len(weights)

    # initialise output
    _X = np.zeros((num_examples, num_kernels * 2), dtype=np.float32) # 2 features per kernel

    for i in range(num_examples):

        for j in prange(num_kernels):

            _X[i, (j * 2):((j * 2) + 2)] = \
            apply_kernel(X[i], weights[j][:lengths[j]], lengths[j], biases[j], dilations[j], paddings[j])

    return _X