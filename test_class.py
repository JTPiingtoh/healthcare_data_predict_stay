import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors._base import NeighborsBase
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist


class PRKNeighborsClassifier(ClassifierMixin, NeighborsBase, BaseEstimator):
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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse=False
        return tags
    
    # fit knn model
    def fit(self, X, y):
        '''
        Calls sklearn's fit method as well as setting some convenience variables
        '''
        
        self._knn_model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            
            algorithm = self.algorithm,
            leaf_size = self.leaf_size,
            metric = self.metric,
            metric_params = self.metric_params,
            p = self.p,
            n_jobs = self.n_jobs
        ) 
        self._knn_model.fit(X,y)
        self.is_fitted_ = True
        X, y = validate_data(self, X, y)
        self.X_ = X
        self.y_ = y
        self.classes_ = unique_labels(y)
        self._class_radii = self._get_class_radii()
        self._proximal_ratios = self._get_proximal_ratios()

        return self


    def _get_class_radii(self):
        '''
        Get the class radii, stored in a dict.
        '''   

        X, y = self.X_, self.y_
        target_classes = self.classes_
        class_radii = {}

        for target_class in target_classes:

            class_rows = X[y == target_class]
            rows = class_rows.shape[0]
            pairwise_distances = pdist(class_rows, metric='euclidean')
            class_radii[target_class] = np.mean(pairwise_distances)

        return class_radii
    

    def _get_proximal_ratios(self):
        
        X, y = self.X_, self.y_


        proximal_ratios = np.empty((X.shape[0], ), dtype='float64')

        for id, x in enumerate(X):
            radius = self._class_radii[y[id]]

            distances, knn_indices = self._knn_model.kneighbors(x.reshape(1, -1), n_neighbors=self.n_neighbors)

            target_class = y[id]
            knn_classes = y[knn_indices.reshape(-1,)]

            val = np.sum( 
                    1.0*(
                        (distances < radius) & (target_class == knn_classes)
                        )
                )

            c = np.sum(1.0*(distances < radius))

            # Handle 0. If c is 0, so is val, which would suggest that the point is an outlier within its own class, therefore has a proximal ratio of 0.
            if c == 0:
                proximal_ratios[id] = 0.0

            else:
                proximal_ratios[id] = val / c

        return np.array(proximal_ratios)


    def predict(self, X, y=None):

        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        distances, indexes = self._knn_model.kneighbors(X, n_neighbors=self.n_neighbors)
        scores = distances / self._proximal_ratios[indexes]
        
        y_pred = np.empty((X.shape[0],), dtype=X.dtype)

        # TODO: implement in cpp
        # assign label of class with max weight
        for id, d in enumerate(scores):
            # get classes
            
            id: int = id

            classes = self.y_[indexes[id]]
            # print(classes)
            # for each class c present in d, find weight of class
            unique_classes = np.unique(classes)
            average_weights = np.zeros(unique_classes.shape)
            
            for j, clss in enumerate(unique_classes):
                average_weights[j] = np.mean(d[classes == clss])
            print(y_pred)
            
            y_pred[id] = unique_classes[np.argmax(average_weights)]

        return y_pred
            