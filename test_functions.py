from scipy.spatial.distance import euclidean

import numpy as np
import pandas as pd


def get_class_average_eu_distance(vals: pd.DataFrame, target: str) -> list:
    '''
    Takes training data, and produces a list containing the average paiwise l2 norm for all points within each target class.
    Data must be tranformed before use. 

    Parameters
    ----------

    df : pd.Dataframe
        Input array or dataframe
    target : str
        String containing the name of the target column 
    
    '''

    target_classes = np.unique(vals[:,-1])
    # drop target column
    x_vals = np.delete(vals, -1,1)
    class_mean_distances = []

    for t_class in target_classes:
        
        # compute class pairwise euclidean radius (mean average)

        mean_distance = 0
        n = 1

        # get rows where target == t_class
        class_values = x_vals[vals[:,-1] == t_class]
        rows = class_values.shape[0]

        for i in range(rows):
            for j in range(i + 1, rows):
                
                distance_eu = euclidean(x_vals[i], x_vals[j])
                mean_distance = ( (mean_distance * n) + distance_eu ) / (n + 1)
                n+=1

        class_mean_distances.append(mean_distance)

    
    return class_mean_distances

if __name__ == "__main__":
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
    from sklearn.model_selection import train_test_split

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer # only using for test purposes

    df = pd.read_csv('Heart.csv')
    ordinal_columns = [
        'RestECG',
        'Slope',
        'Ca'
    ]

    contineous_columns = [
        'Age',
        'RestBP'
        'Chol',
        'MaxHR',
        'Oldpeak'
    ]

    categoric_columns = [
        'Sex',
        'ChestPain',
        'Fbs',
        'ExAng',
        'Thal'
    ]

    encoder = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categoric_columns),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_columns),
        
        ]
    )

    # for testing on test data
    knn_pipe_list = [
        ('encoder', encoder),
        ('scaler', MinMaxScaler()),
        ('imputer', SimpleImputer()),
        # ('classifier', KNeighborsClassifier(n_neighbors=5))
    ]

    knn_pipe_list_classifier = [
        ('encoder', encoder),
        ('scaler', MinMaxScaler()),
        ('imputer', SimpleImputer()),
        ('classifier', KNeighborsClassifier(n_neighbors=5))
    ]

    knn_pipe = Pipeline(knn_pipe_list)
    knn_pipe_classifier = Pipeline(knn_pipe_list_classifier) # with classifier
    target = 'AHD'
    X = df.drop(target, axis=1)
    y = df[target]

    y = 1*(y == 'Yes')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
    encoded_x_train = knn_pipe.fit_transform(X_train)

    # issue is target index will change post transformation: kist use dataframes

    # print(get_class_average_eu_distance(encoded_x_test, target))
    # print(encoded_x_test)

    from test_class import PRKNeighborsClassifier

    prknn = PRKNeighborsClassifier(n_neighbors=5)
    prknn.fit(encoded_x_train, y_train)

    prknn._get_class_radii()

    # print(y_train)
    # print(prknn._class_radii)
    print(prknn._get_proximal_ratios())