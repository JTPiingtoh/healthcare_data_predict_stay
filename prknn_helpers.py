import numpy as np
import numba
from scipy.spatial.distance import pdist


@numba.jit()
def get_class_radii_euclidean(X: np.array, y: np.array, target_classes: np.array):
    '''
    Get the class radii, stored in a dict.
    '''   
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