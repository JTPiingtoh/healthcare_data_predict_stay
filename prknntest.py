import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # only using for test purposes

import numpy as np

from test_class import PRKNeighborsClassifier
from test_class2 import PRKNeighborsClassifier2
from sklearn.utils.estimator_checks import check_estimator

check_estimator(PRKNeighborsClassifier())


