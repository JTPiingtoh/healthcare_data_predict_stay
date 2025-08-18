import numpy as np
from sklearn.utils.extmath import weighted_mode

_y = np.array([1,0,1,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,1,0,0,1,0,1,0,1,0,1,0]).reshape((-1, 1))
classes = [0,1,2]
cls =  np.array([0,0,1,2,1,2]).reshape(1,-1)
dist = np.array([1,0,0,0,5,0], dtype=np.float64).reshape(1,-1)

with np.errstate(divide='ignore'):
    dist = 1 / dist

inf_mask = np.isinf(dist)
inf_row = np.any(inf_mask, axis=1)

# infs are now set to 1, all else are set to 0.
dist[inf_row] = inf_mask[inf_row]

weights = dist

if weights is not None:
	mode, _ = weighted_mode(cls, weights, axis=1)

print(mode)
