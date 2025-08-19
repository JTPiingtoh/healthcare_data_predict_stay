import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors._base import NeighborsBase
from sklearn.utils.validation import validate_data, check_is_fitted, check_array, _num_samples, check_X_y
from sklearn.utils.multiclass import check_classification_targets
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

        prversion="standard" 
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
        self.weights=weights
        self.prversion = prversion


    # These need to be set to bypass certain checks 
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse=False
        tags.target_tags.multi_output=False

        # poor_score is defined as anything below 0.83. However, seems like a harsh metric when using for higher ordinality targets.
        tags.classifier_tags.poor_score=False
        return tags
    
    # fit knn model and calculate proximal ratio
    def fit(self, X, y):
        '''
        Calls sklearn's fit method as well as setting some convenience variables
        '''

        if self.prversion not in ["standard", "enhanced", "weighted"]:
            raise ValueError("prversion not recognised; must be either 'standard', 'enhanced' or 'weighted'")

        # Multi output not allowed
        X, y = validate_data(
                    self,
                    X,
                    y,
                    accept_sparse=False,
                    multi_output=False,
                )
        
        check_classification_targets(y) 
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        
        self._knn_model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            algorithm = self.algorithm,
            leaf_size = self.leaf_size,
            metric = self.metric,
            metric_params = self.metric_params,
            p = self.p,
            n_jobs = self.n_jobs
        ) 
        
        # fit internal knn model
        self._knn_model.fit(X,y)
        self._class_radii = self._get_class_radii()
        self._proximal_ratios = self._get_proximal_ratios()
        self.is_fitted_ = True

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
            pairwise_distances = pdist(class_rows, metric='euclidean')
            class_radii[target_class] = np.mean(pairwise_distances)

        return class_radii
    

    def _get_proximal_ratios(self):
        
        X, y = self.X_, self.y_

        proximal_ratios = np.empty((X.shape[0], ), dtype='float64')

        for id, x in enumerate(X):
            # _class_radii is a dict
            target_class = y[id]
            radius = self._class_radii[target_class]

            distances, knn_indices = self._knn_model.kneighbors(x.reshape(1, -1), n_neighbors=self.n_neighbors)

            
            knn_classes = y[knn_indices.reshape(-1,)]

            val = np.sum( 
                    1.0*(
                        (distances < radius) & (target_class == knn_classes)
                        )
                )

            c = np.sum(1.0*(distances < radius))

            # print(distances, radius)
            # Handle 0. If c is 0, so is val, which would suggest that the point is an outlier within its own class, 
            # therefore has a proximal ratio of 0.
            if c == 0:
                proximal_ratios[id] = 0.0

            else:
                proximal_ratios[id] = val / c

        return np.array(proximal_ratios)


    def predict(self, X, y=None):

        check_is_fitted(self)
        
        X = validate_data(self, X, reset=False)
        n_outputs = len([self.classes_])
        n_queries = _num_samples(self._fit_X if X is None else X)

        distances, indexes = self._knn_model.kneighbors(X, n_neighbors=self.n_neighbors)
        # print(distances, indexes)

        version = self.prversion

        # Divide by zero warning removed, as inf value is valid of weighted mode calculation
        with np.errstate(
            divide="ignore"
        ):
            if version == "standard":
                ww = self._proximal_ratios[indexes] / distances
            elif version == "enhanced":
                ww = self._proximal_ratios[indexes] 
            elif version == "weighted":
                

        # print(ww)
        y_pred = np.empty((X.shape[0],), dtype=self.classes_[0].dtype)

        # TODO: implement in cpp
        # assign label of class with max weight
        for id, weights in enumerate(ww):
            # get classes
            
            id: int = id
        
            # the classes of each nieghbor
            classes = self.y_[indexes[id]]

            # the unique classes 
            fitted_classes = self.classes_

            average_weights = np.zeros(fitted_classes.shape, dtype="float64")
            
            for j, clss in enumerate(fitted_classes):
                # This will cause problems with targets with more than 2 possible 
		        # classes: any class with even 1 inf score will lead to an inf average,
		        # leading to multiple classes with an inf weight

                # TODO: If no neighbours are clss, average_weights[j] will == nan. 
                # Add check, and if class is not present, set weight to 0.
                if not np.any([classes == clss]):
                    average_weights[j] = 0
                else:
                    average_weights[j] = np.mean(weights[classes == clss])
            
            # print(average_weights, classes, fitted_classes[np.argmax(average_weights)]) 
            
            y_pred[id] = fitted_classes[np.argmax(average_weights)]

        #TODO: change this to skl's standard implementation?
        return y_pred
            
