import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys

# CUR_FILE_DIR = os.path.dirname(sys.argv[0])
CUR_FILE_DIR = os.path.dirname(__file__)

# convert flower names to numeric values
def type_to_numeric(x: str) -> int:
    if x == "setosa":
        return 0
    if x == "versicolor":
        return 1
    else:
        return 2
    
# read data and process
def get_data() -> tuple[np.ndarray, np.ndarray]:

    iris_path = os.path.join(CUR_FILE_DIR, "iris.csv")
    print(f"Opening data file: {iris_path}")
    iris_data = pd.read_csv(iris_path)


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

    print(f"X_np: {X_np.shape} {type(X_np)}\n{X_np[0:5, :]}")
    print(f"Y_np: {Y_np.shape} {type(Y_np)}\n{Y_np[0:5]}")

    # create scaler fitting our feature data
    scaler = StandardScaler().fit(X_np)

    # apply the scaling on our data
    X_np = scaler.transform(X_np)

    # lets do one-hot-encoding for our output labels
    Y_np = tf.keras.utils.to_categorical(Y_np, 3)

    # return feature and label data
    return X_np, Y_np

def base_model_config():
    model_config = {
            "HIDDEN_NODES" : [32,64],
            "HIDDEN_ACTIVATION" : "relu",
            "OUTPUT_NODES" : 3,
            "OUTPUT_ACTIVATION" : "softmax",
            "WEIGHTS_INITIALIZER" : "random_normal",
            "BIAS_INITIALIZER" : "zeros",
            "NORMALIZATION" : "none",
            "OPTIMIZER" : "rmsprop",
            "LEARNING_RATE" : 0.001,
            "REGULARIZER" : None,
            "DROPOUT_RATE" : 0.0,
            "EPOCHS" : 10,
            "BATCH_SIZE" : 16,
            "VALIDATION_SPLIT" : 0.2,
            "VERBOSE" : 0,
            "LOSS_FUNCTION" : "categorical_crossentropy",
            "METRICS" : ["accuracy"]
            }
    return model_config



if __name__ == "__main__":

    accuracy_measures = {}

    for batch_size in range(16, 128, 16):

        # load config using defaults
        model_config = base_model_config()