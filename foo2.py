import numpy as np
from sklearn.utils.extmath import weighted_mode
from sklearn.metrics import pairwise_distances_chunked
import numba
from scipy.spatial.distance import pdist
import gc
import numba as nb

from prknn_helpers import get_class_radii_euclidean_non_vectorized

def reduce_func(D_chunk, _):
    mask = ~np.triu(np.ones_like(D_chunk, dtype=bool))
    mean = np.mean(D_chunk[mask])
    return np.full(shape = (D_chunk.shape[0],), fill_value=mean)


n_observations_X = 50000
n_feilds = 2


np.random.seed(33)
arr = np.random.randn(n_observations_X, n_feilds)

mean_d = get_class_radii_euclidean_non_vectorized(arr)

mean_d = np.array([mean_d])
np.savetxt("mean_d", mean_d)
    