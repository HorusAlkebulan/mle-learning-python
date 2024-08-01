from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# import sys
from tensorflow.keras.optimizers import Adagrad, RMSprop, Adam, SGD

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

def get_optimizer(optimizer_name: str, learning_rate: float):
    optimizer = None

    if optimizer_name == 'adagrad':
        optimizer = Adagrad(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate)

    return optimizer

def create_and_run_model(model_config, X, Y, model_name):
    
    model = tf.keras.models.Sequential(name=model_name)

    for layer in range(len(model_config["HIDDEN_NODES"])):

        if (layer == 0):
            model.add(
                keras.layers.Dense(
                    model_config["HIDDEN_NODES"][layer],
                    input_shape=(X.shape[1],),
                    name="Dense-Layer-" + str(layer),
                    kernel_initializer=model_config["WEIGHTS_INITIALIZER"],
                    bias_initializer=model_config["BIAS_INITIALIZER"],
                    kernel_regularizer=model_config["REGULARIZER"],
                    activation=model_config["HIDDEN_ACTIVATION"]),
            )
        else:
            if (model_config["NORMALIZATION"] == "batch"):
                model.add(keras.layers.BatchNormalization())
            
            if (model_config["DROPOUT_RATE"] > 0.0):
                model.add(keras.layers.Dropout(model_config["DROPOUT_RATE"]))

            model.add(
                keras.layers.Dense(
                    model_config["HIDDEN_NODES"][layer],
                    name="Dense-Layer-" + str(layer),
                    kernel_initializer = model_config["WEIGHTS_INITALIZER"],
                    bias_initializer = model_config["BIAS_INITIALIZER"],
                    kernel_regularizer = model_config["REGULARIZER"],
                    activation = model_config["HIDDEN_ACTIVATION"],
                )
            )

        model.add(
            keras.layers.Dense(
                model_config["OUTPUT_NODES"],
                name="Output-layer",
                activation=model_config["OUTPUT_ACTIVATION"],
                )
        )

        optimizer = get_optimizer(
            model_config["OPTIMIZER"],
            model_config["LEARNING_RATE"],
        )

        model.compile(
            loss=model_config["LOSS_FUNCTION"],
            optimizer=optimizer,
            metrics=model_config["METRICS"],
        )

        print("\n------------------------------------------")
        print(model.summary())

        X_train, X_val, Y_train, Y_val = train_test_split(
            X, 
            Y,
            stratify=Y,
            test_size=model_config["VALIDATION_SPLIT"]
        )

        history = model.fit(
            X_train,
            Y_train,
            batch_size=model_config["BATCH_SIZE"],
            epochs=model_config["EPOCHS"],
            verbose=model_config["VERBOSE"],
            validation_data=(X_val, Y_val),
        )

        return history

def plot_graph(accuracy_measures, title):

    plt.figure(figsize=(15,8))
    
    for experiment in accuracy_measures.keys():
        plt.plot(accuracy_measures[experiment], label=experiment, linewidth=3)

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plot_path = title + ".png"
    plot_path = os.path.join(CUR_FILE_DIR, plot_path)
    print(f"Saving plot to {plot_path}")
    plt.savefig(plot_path)

# if __name__ == "__main__":

#     # initialize the measures dict that stores training histories
#     batch_size_epoch_experiment(get_data, base_model_config, create_and_run_model, plot_graph)