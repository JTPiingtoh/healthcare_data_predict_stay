import numpy as np
from sklearn.utils.extmath import weighted_mode
import numba

print(np.__version__)

arr = np.array([
    [1,2],
    [3,4],
    [3,4],
    [3,4],
    [3,4],
    [3,4],
    [3,4],
    [5,6]
], dtype='float64')

@numba.jit()
def pair_distances_mean(arr: np.array):

    mean = 0
    n = 0

    for i in range(arr.shape[0]):
        for j in range(i+1, arr.shape[0]):
            distance = np.linalg.norm(arr[i] - arr[j])
            mean = (mean * n + distance) / (n + 1)
            n+=1
    
    return mean

print(pair_distances_mean(arr))
