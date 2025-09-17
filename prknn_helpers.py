import numpy as np
import numba
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances_chunked

@numba.jit()
def get_class_radii_euclidean_non_vectorized(X: np.array, y: np.array, target_classes: np.array):

    class_radii = np.empty(target_classes.shape)

    for i, target_class in enumerate(target_classes):

        class_rows = X[y == target_class]

        mean = 0
        n = 0

        for j in range(class_rows.shape[0]):
            for k in range(j+1, class_rows.shape[0]):
                pair_distance = np.linalg.norm(class_rows[j] - class_rows[k])
                mean = (mean * n + pair_distance) / (n + 1)
                n+=1

        class_radii[i] = mean

    # for target_class in target_classes:

    #     class_rows = X[y == target_class]
    #     pairwise_distances = pdist(class_rows, metric='euclidean')
    #     class_radii[target_class] = np.mean(pairwise_distances)

    return class_radii


def get_mean_euclidean_chunked(X: np.array):
    '''
    Calculated the mean pairwise distances using chunking
    '''

    D_chunks = pairwise_distances_chunked(X)

    total_size = 0
    total_sum = 0
    
    for D_chunk in D_chunks:
        mask = ~np.triu(np.ones_like(D_chunk, dtype=bool))
        total_size += D_chunk[mask].shape[0]
        total_sum += np.sum(D_chunk[mask])


    return total_sum / total_size




if __name__ == "__main__":

    n_observations_X = 5000_0
    n_feilds = 2

    arr = np.random.randn(n_observations_X, n_feilds)



    print(get_mean_euclidean_chunked(arr))