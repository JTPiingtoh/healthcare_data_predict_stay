import numpy as np

class PRKNN_kwarg_handler():
    '''
        Handles key word arguments for the PRKNN class. THe PRKNN effecitvely wraps 2 knn models implemented by scikit-learn;
        one for calculating proximal ratios (pr_knn), and one for predicting (predict_knn). This argument handeler allows the 
        caller to set the kwargs of both models to be identical (the default) by setting pr-eq_predict to True, or to have different kwargs. 

        Convention: the pr_knn is considered to be the default for kwarg setting for both models.    
    '''

    def __init__(
        self,
        pr_version: str | None = "standard", 
        base_knn_params: dict | None = None,
        pr_knn_params: dict | None = None,
        predict_knn_params: dict | None = None
    ):        
        
        if base_knn_params and (pr_knn_params or predict_knn_params):
            raise ValueError(f"pr and predict knn parameters can not be set alongside base parameters")

        if (pr_knn_params and not predict_knn_params) or (not pr_knn_params and predict_knn_params):
            raise ValueError(f"both pr and predict knn parameters must be set if base parameters not set.")


        self.base_knn_params = base_knn_params
        self.pr_knn_params = pr_knn_params
        self.predict_knn_params = predict_knn_params
        self.pr_version = pr_version

    def _fit_params(self):

        # used in _generate_kwargs_for_knn()
        self._knn_kwargs_list = [
            "n_neighbors",
            "weights",
            "algorithm",
            "leaf_size",
            "p",
            "metric",
            "metric_params",
            "n_jobs",
        ]

        defaults_base = dict(
                n_neighbors=5,
                weights="uniform",
                algorithm="auto",
                leaf_size=30,
                p=2,
                metric="minkowski",
                metric_params=None,
                n_jobs=None,
            )
        
        defaults = {}
        if self.base_knn_params:
            defaults = defaults_base
        else:
            for key, value in defaults_base.items():
                defaults["pr_" + key] = value
                defaults["predict_" + key] = value
        
        pr_valid_kwargs = []
        predict_valid_kwargs = []
        valid_kwargs = []
        if self.base_knn_params:
            valid_kwargs = [argument for argument in defaults_base]
        else:
            pr_valid_kwargs = ["pr_" + argument for argument in defaults_base]
            predict_valid_kwargs = ["predict_" + argument for argument in defaults_base]
            valid_kwargs = predict_valid_kwargs + pr_valid_kwargs

        merged = defaults
        if self.base_knn_params:
            merged = {**merged, **self.base_knn_params}

        if self.pr_knn_params:
            for argument in self.pr_knn_params:
                if argument in pr_valid_kwargs:
                    continue
                raise ValueError(f"{argument} passed to pr_knn_params: Expected {pr_valid_kwargs}")

            merged = {**merged, **self.pr_knn_params}

        if self.predict_knn_params:
            for argument in self.predict_knn_params:
                if argument in predict_valid_kwargs:
                    continue
                raise ValueError(f"{argument} passed to predict_knn_params: Expected {predict_valid_kwargs}")


            merged = {**merged, **self.predict_knn_params}

        
        for key, value in merged.items():
            if key not in valid_kwargs:
                raise ValueError(f"Got unexpected argument {key}, expected {valid_kwargs}")
            elif self.base_knn_params:
                setattr(self,"_pr_" + key,value)
                setattr(self,"_predict_" + key,value)
            else:
                setattr(self,"_" + key,value)

            

        self._params_fitted = True


    def _generate_kwargs_for_knn(self, knn_model_prefix:str, **kwargs):

        assert self._params_fitted

        assert(knn_model_prefix in ["pr_", "predict_"])

        # for checking weights is supplied when using weighted predict_knn
        if self.pr_version == "weighted" and knn_model_prefix == "predict_":
            assert("weights" in kwargs.keys())

        kwags_dict = {}

        # if self.pr_eq_predict:
        #     prefix = ""
        # else:
        prefix = "_" + knn_model_prefix

        for argument in self._knn_kwargs_list:

            if argument in kwargs:
                kwags_dict[argument] = kwargs[argument]
            else:
                kwags_dict[argument] = vars(self)[prefix + argument]



        return(kwags_dict)


if __name__ == "__main__":

    from sklearn.neighbors import KNeighborsClassifier

    pr_params = {
        "pr_n_jobs": 6,
        "pr_n_neighbors": 11,
        "pr_weights": "FOO"
        }
    
    predict_params = {
        "predict_n_jobs": 8
    }

    handler1 = PRKNN_kwarg_handler(pr_knn_params=pr_params, predict_knn_params=predict_params)
    handler1._fit_params()

    for key, value in vars(handler1).items():
        print(key, value)

    print(handler1._generate_kwargs_for_knn("pr_", weights="FOO"))

    # testknn = KNeighborsClassifier(handler1._generate_kwargs_for_knn("pr_"))
