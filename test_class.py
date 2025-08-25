# TODO: fix tuple issue with member vars when initing internal knn model, then continue to split prknn and predict knn.


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors._base import NeighborsBase
from sklearn.utils.validation import validate_data, check_is_fitted, check_array, _num_samples, check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist


class PRKNeighborsClassifier(ClassifierMixin, BaseEstimator):
    '''
    Extends sklearn's KNeighborsClassifier by implementing proximal ratio weights as proposed by 
    https://journalofbigdata.springeropen.com/articles/10.1186/s40537-025-01137-2

    Note: weights can only be supplied when not using the "weighted" prKNN, which uses proximal ratios as weights.
    '''
    def __init__(
        self,
        prversion="standard", 
        pr_n_neighbors=5,
        *,
        pr_weights="uniform",
        pr_algorithm='auto',
        pr_leaf_size=30,
        pr_p=2,
        pr_metric="minkowski",
        pr_metric_params=None,
        pr_n_jobs=None,

        predict_eq_pr=True,

        predict_n_neighbors=5,
        predict_weights="uniform",
        predict_algorithm='auto',
        predict_leaf_size=30,
        predict_p=2,
        predict_metric="minkowski",
        predict_metric_param=None,
        predict_n_jobs=None,


    ):
        # TODO: Need to seperate pr model and prediction model's kwargs somehow.
        
        self.prversion = prversion

        self.pr_n_neighbors=pr_n_neighbors
        self.pr_algorithm=pr_algorithm
        self.pr_leaf_size=pr_leaf_size
        self.pr_weights = pr_weights
        self.pr_metric=pr_metric
        self.pr_p=pr_p
        self.pr_metric_params=pr_metric_params
        self.pr_n_jobs=pr_n_jobs

        self.predict_eq_pr=predict_eq_pr

        valid_kwargs = [
            "n_neighbors", "weights", "algorithm", "leaf_size", "p",
            "metric", "metric_params", "n_jobs",
        ]

        

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
        Computes the training set's proximal ratios, and fits the prediction model ready for prediciton.
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
        
        self._pr_knn_model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            algorithm = self.algorithm,
            leaf_size = self.leaf_size,
            metric = self.metric,
            metric_params = self.metric_params,
            p = self.p,
            n_jobs = self.n_jobs,
            weights=self.weights
        )
        
        # fit internal knn model
        self._pr_knn_model.fit(X,y)
        self._class_radii = self._get_class_radii()
        self._proximal_ratios = self._get_proximal_ratios()
        self.is_fitted_ = True
        if self.prversion == "weighted":
            prediction_model_weights = self._return_proximal_ratios
        else: 
            prediction_model_weights = self.prediction_model_weights 

        self._prediction_knn_model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=prediction_model_weights).fit(X,y)

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

            distances, knn_indices = self._pr_knn_model.kneighbors(x.reshape(1, -1), n_neighbors=self.n_neighbors)

            
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


    def _return_proximal_ratios(self, X): 
        return self._proximal_ratios


    def predict(self, X, y=None):

        check_is_fitted(self)
        version = self.prversion
        X = validate_data(self, X, reset=False)
        weights = self._return_proximal_ratios if version == "weighted" else "uniform"

        # TODO: add weighted model if weighted pr knn being used
        if self.prversion == "weighted":
            distances, indexes = self._prediction_knn_model.kneighbors(X, n_neighbors=self.n_neighbors)
        else:
            distances, indexes = self._pr_knn_model.kneighbors(X, n_neighbors=self.n_neighbors)
        # print(distances, indexes)

        # Divide by zero warning removed, as inf value is valid of weighted mode calculation
        with np.errstate(
            divide="ignore"
        ):
            if version == "standard":
                ww = self._proximal_ratios[indexes] / distances
            elif version == "enhanced":
                ww = self._proximal_ratios[indexes] 
            elif version == "weighted":
                ww = distances

        # print(ww)
        y_pred = np.empty((X.shape[0],), dtype=self.classes_[0].dtype)

        # TODO: implement in cpp
        # assign label of class with max weight
        for query_index, query_weights in enumerate(ww):
     
            # the classes of each nieghbor
            query_classes = self.y_[indexes[query_index]]

            # the unique classes 
            fitted_classes = self.classes_

            class_weights = np.zeros(fitted_classes.shape, dtype="float64")

            for j, clss in enumerate(fitted_classes): 

                # A if class is not present, set weight to 0.
                if not np.any([query_classes == clss]):
                    class_weights[j] = 0
                # TODO: implement weighted prKNN quary weights here
                elif self.prversion == "weighted":
                    # if using weighted prknn, weights at this point are just distances
                    class_distances = query_weights[query_classes == clss]
                    weight_1 = np.sum(1 / class_distances)
                    weight_2 = np.sum([query_classes == clss] * 1) / self.n_neighbors 

                    dist_xk = np.max(class_distances)
                    dist_x1 = np.min(class_distances)
                    if dist_xk == dist_x1:
                        # each point has weight 1
                        weight_3 = np.sum([query_classes == clss] * 1)
                    else:
                        weight_3 = np.sum(
                            ( (dist_xk - dist_x1) / (dist_xk - class_distances) ) * ( (dist_xk + dist_x1) / (dist_xk + class_distances) )
                        )

                    class_weights[j] = weight_1 + weight_2 + weight_3
                    
                else:
                    class_weights[j] = np.mean(query_weights[query_classes == clss])
            
            # print(class_weights, classes, fitted_classes[np.argmax(class_weights)]) 
            
            y_pred[query_index] = fitted_classes[np.argmax(class_weights)]

        #TODO: change this to skl's standard implementation?
        return y_pred
            
