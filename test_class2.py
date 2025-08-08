import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist


class PRKNeighborsClassifier2(KNeighborsClassifier):
    '''
    Extends sklearn's KNeighborsClassifier by implementing proximal ratio weights as proposed by 
    https://journalofbigdata.springeropen.com/articles/10.1186/s40537-025-01137-2
    '''
    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.weights = weights


    def foo(self):
        return self._fit_X
            