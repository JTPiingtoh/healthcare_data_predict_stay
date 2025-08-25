class foo():
    '''
    Extends sklearn's KNeighborsClassifier by implementing proximal ratio weights as proposed by 
    https://journalofbigdata.springeropen.com/articles/10.1186/s40537-025-01137-2

    Note: weights can only be supplied when not using the "weighted" prKNN, which uses proximal ratios as weights.
    '''
    def __init__(
        self,
        pr_version:str = "standard",
        pre_eq_predict: bool = True,
        **kwargs
    ):


        valid_knn_kwargs_base = [
            "n_neighbors", "weights", "algorithm", "leaf_size", "p",
            "metric", "metric_params", "n_jobs",
        ]


        valid_knn_kwargs = []
        
        for kwarg in valid_knn_kwargs_base:
            valid_knn_kwargs.append("pr_" + kwarg)
            valid_knn_kwargs.append("predict_" + kwarg)

        defaults = dict(
                n_neighbors=5,
                weights="uniform",
                algorithm="auto",
                leaf_size=30,
                p=2,
                metric="minkowski",
                metric_params=None,
                n_jobs=None,
                # prediction_model_weights="uniform",
                # prversion="standard",
            )
        
        merged = {**defaults,**kwargs}

        for k in kwargs:
            if k not in valid_knn_kwargs:
                raise ValueError(f"Recieved unexptected argument: {k}")
        
        if not pre_eq_predict:
            for key, val in merged.items():
                setattr(self, key, val)
        else:            
            pairs = float(len(valid_knn_kwargs)) / 2
            assert(pairs % 2 == 0)
            for i in range(int(pairs)):
                i*=2
                setattr(self, valid_knn_kwargs[i], merged[valid_knn_kwargs[i]])
                setattr(self, valid_knn_kwargs[i+1], merged[valid_knn_kwargs[i+1]])


f= foo(
    pr_version="standard",
    pre_eq_predict=True
    )