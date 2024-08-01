# nodes_per_layer_experiment.py

from common_experiment_functions import base_model_config, create_and_run_model, get_data, plot_graph

accuracy_measures = {}

for num_nodes in range(8, 40, 8):

    layer_list = []
    num_layers = 2
    for layer_count in range(num_layers):
        layer_list.append(num_nodes)

    model_config = base_model_config()
    model_config["HIDDEN_NODES"] = layer_list
    X, Y = get_data()
    model_name = f"Nodes-{num_nodes}"

    history = create_and_run_model(
        model_config,
        X,
        Y,
        model_name,
    )

    accuracy_measures[model_name] = history.history["accuracy"]

plot_graph(accuracy_measures, "Varying Number of Nodes per Layer")