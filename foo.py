import numpy as np
from sklearn.utils.extmath import weighted_mode
from sklearn.metrics import pairwise_distances_chunked
import numba
from scipy.spatial.distance import pdist
from math import comb

rows = 10
columns = 2

# TODO: if not columns not divisable by chunks, handle remainder
chunks = 5

arr = np.random.randn(rows,columns)


chunked_means = np.empty((chunks,))
print(chunked_means)

arr2 = arr.reshape(chunks,-1,columns)

for i in range(chunks):
    pairwise_distances = pdist(arr2[i], metric='euclidean')
    chunked_means[i] = np.mean(pairwise_distances)

print(chunked_means)

print(np.mean(chunked_means))
print(np.mean(pdist(arr, 'euclidean')))