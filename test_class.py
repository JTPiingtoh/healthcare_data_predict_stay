# TODO: fix tuple issue with member vars when initing internal knn model, then continue to split prknn and predict knn.


import numpy as np
import numba
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors._base import NeighborsBase
from sklearn.utils.validation import validate_data, check_is_fitted, check_array, _num_samples, check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist

from PRKNN_handler import PRKNN_kwarg_handler
from prknn_helpers import get_mean_euclidean_chunked, _predict_on_weights

from tqdm import tqdm

class PRKNeighborsClassifier(ClassifierMixin, BaseEstimator, PRKNN_kwarg_handler):
    '''
    Extends sklearn's KNeighborsClassifier by implementing proximal ratio weights as proposed by 
    https://journalofbigdata.springeropen.com/articles/10.1186/s40537-025-01137-2

    Note: weights can only be supplied when not using the "weighted" prKNN, which uses proximal ratios as weights.
    '''
    def __init__(
        self,
        pr_version="standard", 
        base_knn_params: dict | None = None,
        pr_knn_params: dict | None = None,
        predict_knn_params: dict | None = None

    ):
        # TODO: Need to seperate pr model and prediction model's kwargs somehow.
        super().__init__(
            pr_version=pr_version,
            base_knn_params = base_knn_params,
            pr_knn_params = pr_knn_params,
            predict_knn_params = predict_knn_params
        )

        self.pr_version = pr_version
   

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

        X = np.array(X, dtype ='float64')
        y = np.array(y, dtype ='float64')

        if self.pr_version not in ["standard", "enhanced", "weighted"]:
            raise ValueError("pr_version not recognised; must be either 'standard', 'enhanced' or 'weighted'")
        
        self._fit_params()

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
        
        # fit pr_knn
        self._pr_knn_model = KNeighborsClassifier(**self._generate_kwargs_for_knn("pr_")).fit(X,y)        

        self._class_radii = self._compute_class_radii()
        self._proximal_ratios = self._compute_proximal_ratios()

        # build prediction model
        if self.pr_version == "weighted":
            # weights must be a callable 
            prediction_model_weights = self._return_proximal_ratios
        else: 
            prediction_model_weights = self._pr_weights 

        self._prediction_knn_model = KNeighborsClassifier(
            **self._generate_kwargs_for_knn(
                knn_model_prefix="predict_", 
                weights=prediction_model_weights)
                ).fit(X,y)
        

        self.is_fitted_ = True


        return self


    def _compute_class_radii(self):
        '''
        Get the class radii, stored in a dict.
        '''   
        # class_radii = {}
        # class_radii_values = get_class_radii_euclidean(X, y, target_classes)

        # for i, radii_value in enumerate(class_radii_values):
        #     class_radii[target_classes[i]] = radii_value  

        X, y = self.X_, self.y_

        target_classes = self.classes_
        class_radii = {}

        for target_class in target_classes:
            
            class_rows = X[y == target_class]
            
            class_radii[target_class] = get_mean_euclidean_chunked(class_rows)

        return class_radii
    

    # TODO: optimize
    def _compute_proximal_ratios(self):
        
        X, y = self.X_, self.y_

        proximal_ratios = np.empty((X.shape[0], ), dtype='float64')

        for id, x in enumerate(X):
            # _class_radii is a dict
            # TODO: change _class_radii to array
            target_class = y[id]
            radius = self._class_radii[target_class]

            distances, knn_indices = self._pr_knn_model.kneighbors(x.reshape(1, -1), n_neighbors=self._pr_n_neighbors)
            
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

        return proximal_ratios


    def _return_proximal_ratios(self): 
        return self._proximal_ratios


    def predict(self, X, y=None):

        check_is_fitted(self)
        version = self.pr_version
        X = validate_data(self, X, reset=False)


        dd, indexes = self._prediction_knn_model.kneighbors(X, n_neighbors=self._predict_n_neighbors)
        
        # print(distances, indexes)

        # Divide by zero warning removed, as inf value is valid of weighted mode calculation
        with np.errstate(
            divide="ignore"
        ):
            if version == "standard":
                ww = self._proximal_ratios[indexes] / dd
            elif version == "enhanced":
                ww = self._proximal_ratios[indexes] 
            elif version == "weighted":
                ww = dd

        # print(ww)
        y_pred = np.empty((X.shape[0],), dtype=self.classes_[0].dtype)

        fitted_classes = self.classes_
        n_neighbors = self._predict_n_neighbors
        y_pred = _predict_on_weights(
            ww=ww,
            y=self.y_,
            indexes=indexes,
            fitted_classes=fitted_classes,
            n_neighbors=n_neighbors,
            version=version,
            y_pred=y_pred
        )

        # TODO: implement in cpp
        # assign label of class with max weight
        for query_index, query_weights in enumerate(ww):
     
            # the classes of each nieghbor
            query_classes = self.y_[indexes[query_index]]

            # the unique classes 

            class_weights = np.zeros(fitted_classes.shape, dtype="float64")

            # TODO: compile
            for j, clss in enumerate(fitted_classes): 

                # A if class is not present, set weight to 0.
                class_mask = query_classes == clss
                if not np.any(class_mask):
                    class_weights[j] = 0
                
                elif version == "weighted":
                    # if using weighted prknn, weights at this point are just distances
                    class_distances = query_weights[class_mask]

                    zero_mask = class_distances == 0
                    if np.any(zero_mask):
                        class_weights[j] = np.inf
                        continue

                    weight_1 = np.sum(1 / class_distances)
                    weight_2 = np.sum(class_mask) / self._predict_n_neighbors 

                    dist_xk = np.max(class_distances)
                    dist_x1 = np.min(class_distances)
                    if dist_xk == dist_x1:
                        # each point has weight 1
                        weight_3 = np.sum(class_mask)
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

        #TODO: change this to skl's standard implementation?
        return y_pred
            
