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


# def get_mean_euclidean_chunked(X: np.array):
#     '''
#     Calculated the mean pairwise distances using chunking
#     '''

#     D_chunks = pairwise_distances_chunked(X)

#     total_size = 0
#     total_sum = 0
    
#     for D_chunk in D_chunks:
#         mask = ~np.triu(np.ones_like(D_chunk, dtype=bool))
#         total_size += D_chunk[mask].shape[0]
#         total_sum += np.sum(D_chunk[mask])


#     return total_sum / total_size


def get_mean_euclidean_chunked(X: np.ndarray) -> float:
    """
    Compute the mean pairwise Euclidean distance using chunking.
    """
    total_sum = 0.0
    total_size = 0

    def reduce_func(D_chunk, start):
        nonlocal total_sum, total_size
        n_samples = D_chunk.shape[1]

        
        row_indices = np.arange(start, start + D_chunk.shape[0])

        for local_row, global_row in enumerate(row_indices):
            
            valid_cols = np.arange(global_row + 1, n_samples)
            dists = D_chunk[local_row, valid_cols]

            total_sum += dists.sum()
            total_size += dists.size

        return None  

    
    list(pairwise_distances_chunked(X, reduce_func=reduce_func))

    return total_sum / total_size

@numba.jit()
def _predict_on_weights(
    ww,
    y,
    indexes,
    fitted_classes,
    n_neighbors,
    version,
    y_pred
    ):

    for query_index, query_weights in enumerate(ww):

        query_classes = y[indexes[query_index]]


        class_weights = np.zeros(fitted_classes.shape, dtype="float64")

        # TODO: compile
        for j, clss in enumerate(fitted_classes): 

            # A if class is not present, set weight to 0.
            class_mask = np.where(query_classes == clss, 1, 0)
            if not np.any(class_mask):
                class_weights[j] = 0
            
            elif version == "weighted":
                # if using weighted prknn, weights at this point are just distances
                class_distances = query_weights[class_mask]

                # If any point in that class has a distance of 0, weight 1 will be infinite, and therefore so will the weight of that entire class
                zero_mask = class_distances == 0
                if np.any(zero_mask):
                    class_weights[j] = np.inf
                    continue

                weight_1 = np.sum(1 / class_distances)
                weight_2 = np.sum(class_mask * 1) / n_neighbors 

                dist_xk = np.max(class_distances)
                dist_x1 = np.min(class_distances)
                if dist_xk == dist_x1:
                    # each point has weight 1
                    weight_3 = np.sum(class_mask * 1)
                else:
                    
                    weight_3 = np.sum(
                        ( (dist_xk - class_distances) / (dist_xk - dist_x1) ) 
                        * 
                        ( (dist_xk + class_distances) / (dist_xk + dist_x1) )
                    )

                class_weights[j] = weight_1 + weight_2 + weight_3
                
            else:
                class_weights[j] = np.mean(query_weights[class_mask])
        
        # print(class_weights, classes, fitted_classes[np.argmax(class_weights)]) 
        
        y_pred[query_index] = fitted_classes[np.argmax(class_weights)]

    return y_pred



if __name__ == "__main__":

    n_observations_X = 5000_0
    n_feilds = 2

    arr = np.random.randn(n_observations_X, n_feilds)



    print(get_mean_euclidean_chunked(arr))