from typing import Union, Tuple

import numpy as np


def ema(values, alpha):
    ema_values = np.zeros_like(values)
    ema_values[0] = values[0]
    for i in range(1, len(values)):
        ema_values[i] = alpha * values[i] + (1 - alpha) * ema_values[i - 1]
    return ema_values


def tensorlist_to_numpy(tensorlist):
    steps = len(tensorlist)
    latents = len(tensorlist[0])
    np_array = np.zeros((steps, latents))
    for i, tensor in enumerate(tensorlist):
        np_array[i, :] = tensor.to('cpu').numpy()
    return np_array
