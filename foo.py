import numpy as np
from sklearn.utils.extmath import weighted_mode
from sklearn.metrics import pairwise_distances_chunked
import numba
from scipy.spatial.distance import pdist
from math import comb
import gc
import numba as nb

def reduce_func(D_chunk, _):
    mask = ~np.triu(np.ones_like(D_chunk, dtype=bool))
    mean = np.mean(D_chunk[mask])
    return np.full(shape = (D_chunk.shape[0],), fill_value=mean)


n_observations_X = 5000_0
n_feilds = 2

# TODO: if not columns not divisable by chunks, handle remainder
chunks = 5

arr = np.random.randn(n_observations_X, n_feilds)

D_chunks = pairwise_distances_chunked(arr)


def grand_mean(D_chunks):
    
    total_size = 0
    total_sum = 0
    
    for D_chunk in D_chunks:
        mask = ~np.triu(np.ones_like(D_chunk, dtype=bool))
        total_size += D_chunk[mask].shape[0]
        total_sum += np.sum(D_chunk[mask])

        print(total_sum)

    return total_sum / total_size

print(grand_mean(D_chunks))