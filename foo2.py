import numpy as np
from sklearn.utils.extmath import weighted_mode
from sklearn.metrics import pairwise_distances_chunked
import numba
from scipy.spatial.distance import pdist
import gc
import numba as nb

def reduce_func(D_chunk, _):
    mask = ~np.triu(np.ones_like(D_chunk, dtype=bool))
    mean = np.mean(D_chunk[mask])
    return np.full(shape = (D_chunk.shape[0],), fill_value=mean)


n_observations_X = 50000
n_feilds = 2

arr = np.random.randn(n_observations_X, n_feilds)

# print(np.mean(pdist(arr)))

D_chunks = pairwise_distances_chunked(arr)

prev = 0
for D_chunk in D_chunks:
    print(D_chunk.shape)
    for row in D_chunk:
        # print(np.argmin(row))
        pass
    