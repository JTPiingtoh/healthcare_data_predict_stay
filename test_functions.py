from scipy.spatial.distance import euclidean

import numpy as np
import pandas as pd

from test_class import PRKNeighborsClassifier
from test_class2 import PRKNeighborsClassifier2
from sklearn.utils.estimator_checks import check_estimator

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # only using for test purposes


#check_estimator(PRKNeighborsClassifier())

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

prknn = PRKNeighborsClassifier(
    pr_version="weighted",
    pr_eq_predict=False
)


knn_pipe_list_classifier = [
    ('encoder', encoder),
    ('scaler', MinMaxScaler()),
    ('imputer', SimpleImputer()),
    ('classifier', prknn)
]

knn_pipe = Pipeline(knn_pipe_list)
knn_pipe_classifier = Pipeline(knn_pipe_list_classifier) # with classifier
target = 'AHD'
X = df.drop(target, axis=1)
y = df[target]

y = 1*(y == 'Yes')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

knn_pipe_classifier.fit(X_train,y_train)

y_pred = knn_pipe_classifier.predict(X_test)
# score = knn_pipe_classifier.score(X_test, y_test)

#print(score)
#print(knn_pipe_classifier["classifier"]._proximal_ratios)
