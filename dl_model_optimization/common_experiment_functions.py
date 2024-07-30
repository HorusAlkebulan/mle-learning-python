import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# convert flower names to numeric values
def type_to_numeric(x: str) -> int:
    if x == "setosa":
        return 0
    if x == "versicolor":
        return 1
    else:
        return 2
    
# read data and process
def get_data():
    iris_data = pd.read_csv("iris.csv")

    print(f"dtypes: \n{iris_data.dtypes}")
    print(f"describe: \n{iris_data.describe()}")
    print(f"head: \n{iris_data.head()}")

    # encode string labels to numeric values
    label_encoder = preprocessing.LabelEncoder()
    iris_data["Species"] = label_encoder.fit_transform(
        iris_data["Species"]
    )

    # convert data to numpy
    iris_np = iris_data.to_numpy()

    # test_common_experiment_functions.py::test_get_data dtypes: 
    # Sepal.Length    float64
    # Sepal.Width     float64
    # Petal.Length    float64
    # Petal.Width     float64
    # Species          object

    # separate feature (first 4 columns) and target (last column)
    X_np = iris_np[:, 0:4]
    Y_np = iris_np[:, -1]

    print(f"X_np: {str(X_np)[0:100]}")
    print(f"Y_np: {str(Y_np)[0:100]}")