# hidden_layers_experiment.py
import os
from common_experiment_functions import base_model_config, create_and_run_model, get_data, plot_graph

PROJECT_ROOT = os.path.dirname(__file__)

accuracy_measures = {}
layer_list = []

for layer_count in range(1, 6):

    num_nodes = 32
    layer_list.append(32)

    model_config = base_model_config()
    X, Y = get_data()

    model_config["HIDDEN_NODES"] = layer_list
    model_name = f"Layers-{layer_count}"

    history = create_and_run_model(
        model_config,
        X,
        Y,
        model_name,
    )

    accuracy_measures[model_name] = history.history["accuracy"]

plot_graph(accuracy_measures, "Varying Number of Hidden Layers")

