import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean



class PRKNeighborsClassifier(BaseEstimator):
    def __init__(
            self,
            n_neighbors
            ):
        self.n_neighbors = n_neighbors
        self.knn_model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        super().__init__()

    # fit knn model
    def fit(self, X, y):
        '''
            Calls sklearn's fit method as well as setting some convenience variables
        '''
        self.knn_model.fit(X,y)
        self._X = X
        self._y = y
        self.classes_ = self.knn_model.classes_
        self.is_fitted_ = True
        

    def _get_class_radii(self):
        '''
        Get the class radii, stored in a dict.
        '''
        if self.is_fitted_ is None:
            raise RuntimeError("KNN model has not been fit when getting class radii.")
        target_classes = self.classes_

        class_radii = {}

        for target_class in target_classes:

            mean_distance = 0
            n = 1

            class_rows = self._X[self._y == target_class]
            rows = class_rows.shape[0]

            for i in range(rows):
                for j in range(i + 1, rows):
                    distance_eu = euclidean(class_rows[i], class_rows[j])
                    mean_distance = ( (mean_distance * n) + distance_eu ) / (n + 1)
                    n+=1

            class_radii[target_class] = mean_distance

        self._class_radii = class_radii


    def _get_proximal_ratios(self):
        
        proximal_ratios = np.empty((self._X.shape[0], ), dtype='float64')

        for id, x in enumerate(self._X):
            radius = self._class_radii[self._y.values[id]]

            distances, knn_indices = self.knn_model.kneighbors(x.reshape(1, -1), n_neighbors=self.n_neighbors)

            target_class = self._y.values[id]
            knn_classes = self._y.values[knn_indices.reshape(-1,)]

            val = np.sum( 
                    1.0*(
                        (distances < radius) & (target_class == knn_classes)
                        )
                )

            c = np.sum(1.0*(distances < radius))

            print(val, c)

            proximal_ratios[id] = val / c

        return proximal_ratios



    def predict(self, X, y=None):
        pass
        # get x_test neighbors
        # self.knn_model.kneighbors(X, n_neighbors=self.n_neighbors)

        # 