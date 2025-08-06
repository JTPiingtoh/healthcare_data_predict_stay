import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist


class PRKNeighborsClassifier(ClassifierMixin, BaseEstimator):
    '''
    Extends sklearn's KNeighborsClassifier by implementing proximal ratio weights as proposed by 
    https://journalofbigdata.springeropen.com/articles/10.1186/s40537-025-01137-2
    '''
    def __init__(
            self,
            n_neighbors
            ):
        self.n_neighbors = n_neighbors
        self.knn_model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.is_fitted_ = False
        super().__init__()

    # fit knn model
    def fit(self, X, y):
        '''
        Calls sklearn's fit method as well as setting some convenience variables
        '''
        self.is_fitted_ = True
        X, y = validate_data(self, X, y)
        self.X_ = X
        self.y_ = y
        self.knn_model.fit(X,y)
        self.classes_ = unique_labels(y)

        if self.classes_ != self.knn_model.classes_:
            raise RuntimeError("Nested knn model has different classes to wrapper model.")

        self._class_radii = self._get_class_radii()
        self._proximal_ratios = self._get_proximal_ratios(X,y)

        return self


    def _get_class_radii(self, X, y):
        '''
        Get the class radii, stored in a dict.
        '''   

        X, y = self.X_.values, self.y_.values
        target_classes = self.classes_
        class_radii = {}

        for target_class in target_classes:

            class_rows = X[y == target_class]
            rows = class_rows.shape[0]
            pairwise_distances = pdist(class_rows, metric='euclidean')
            class_radii[target_class] = np.mean(pairwise_distances)

        return class_radii
    


    def _get_proximal_ratios(self, X, y):
        
        X, y = self.X_.values, self.y_.values

        proximal_ratios = np.empty((X.shape[0], ), dtype='float64')

        for id, x in enumerate(X):
            radius = self._class_radii[y[id]]

            distances, knn_indices = self.knn_model.kneighbors(x.reshape(1, -1), n_neighbors=self.n_neighbors)

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

        return proximal_ratios



    def predict(self, X, y=None):

        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        
        # get x_test neighbors
        # self.knn_model.kneighbors(X, n_neighbors=self.n_neighbors)

        # 