import numpy as np


def sample_from_tuple_or_scalar(rng, x):
    if isinstance(x, tuple):
        return rng.uniform(low=x[0], high=x[1])
    else:
        return x