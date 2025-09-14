import numpy as np
import numba

@numba.jit()
def get_class_radii(X: np.array, y: np.array, target_classes: np.array):
    '''
    Get the class radii, stored in a dict.
    '''   
    class_radii = {}

    for target_class in target_classes:

        class_rows = X[y == target_class]

        mean = 0
        n = 0

        for i in range(class_rows.shape[0]):
            for j in range(i+1, class_rows.shape[0]):
                pair_distance = np.linalg.norm(class_rows[i], class_rows[i])
                mean = (mean * n + pair_distance) / (n + 1)
                n+=1

        class_radii[target_class] = mean

    # for target_class in target_classes:

    #     class_rows = X[y == target_class]
    #     pairwise_distances = pdist(class_rows, metric='euclidean')
    #     class_radii[target_class] = np.mean(pairwise_distances)

    return class_radii