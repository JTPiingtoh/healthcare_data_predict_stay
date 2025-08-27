class PRKNN_kwarg_handler():
    '''
        Handles key word arguments for the PRKNN class. THe PRKNN effecitvely wraps 2 knn models implemented by scikit-learn;
        one for calculating proximal ratios (pr_knn), and one for predicting (predict_knn). This argument handeler allows the 
        caller to set the kwargs of both models to be identical (the default) by setting pr-eq_predict to True, or to have different kwargs. 

        Convention: the pr_knn is considered to be the default for kwarg setting for both models.    
    '''

    def __init__(
        self,
        pr_version:str = "standard",
        pr_eq_predict: bool = True,
        
        **kwargs
    ):

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

        self._pr_version = pr_version
        self._pr_eq_predict = pr_eq_predict


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
        valid_knn_kwargs = []

        for default, value in defaults_base.items():

            if pr_eq_predict == True:
                defaults[default] = value
                valid_knn_kwargs.append(default)

            else:
                defaults["pr_" + default] = value
                defaults["predict_" + default] = value

                valid_knn_kwargs.append("pr_" + default)
                valid_knn_kwargs.append("predict_" + default)

        merged = {**defaults,**kwargs}

        # for argument in self._knn_kwargs_list``:
        #     print(merged[argument])

        for k in kwargs:

            if k not in defaults:
                
                if pr_eq_predict and (k.startswith("pr_") or k.startswith("predict_")): 
                    raise ValueError(f"Recieved argument '{k}': pr_ or predict_ prefix cannot be used for arguments when pr_eq_predict=True")

                elif k in defaults_base:
                    raise ValueError(f"Recieved argument '{k}': pr_ or predict_ prefix must be used for arguments when pr_eq_predict=False")

                else:
                    raise ValueError(f"Recieved unexptected argument: {k}")
        
        if not pr_eq_predict:
            for key, val in merged.items():
                setattr(self, key, val)
        else:            
            pairs = float(len(valid_knn_kwargs)) / 2
            assert(pairs % 2 == 0)
            for i in range(int(pairs)):
                i*=2
                setattr(self, valid_knn_kwargs[i], merged[valid_knn_kwargs[i]])
                setattr(self, valid_knn_kwargs[i+1], merged[valid_knn_kwargs[i+1]])


    def _generate_kwargs_for_knn(self, knn_model_prefix:str, **kwargs):

        assert(knn_model_prefix in ["pr_", "predict_"])

        # for checking weights is supplied when using weighted predict_knn
        if self._pr_version == "weighted":
            assert("weights" in kwargs)

        kwags_dict = {}

        if self._pr_eq_predict:
            prefix = ""
        else:
            prefix = knn_model_prefix

        for argument in self._knn_kwargs_list:

            if argument in kwargs:
                kwags_dict[argument] = kwargs[argument]
            else:
                kwags_dict[argument] = vars(self)[prefix + argument]



        return(kwags_dict)


if __name__ == "__main__":

    from sklearn.neighbors import KNeighborsClassifier

    handler= PRKNN_kwarg_handler(
        pr_version="weighted",
        pr_eq_predict=True,
        n_neighbors=7,
        n_jobs=8
        )

    a = {
        "n_jobs": 6,
        "n_neighbors": 11,
        "weights": "uniform"
        }

    handler1 = PRKNN_kwarg_handler(pr_version="weighted",
                                   pr_eq_predict=True,
                                   **a
    )

    # for key, value in vars(handler1).items():
    #     print(key, value)

    print(handler1._generate_kwargs_for_knn("pr_", weights="FOO"))

    # testknn = KNeighborsClassifier(handler1._generate_kwargs_for_knn("pr_"))
