# activation_function_experiment.py
from common_experiment_functions import get_data, create_and_run_model, base_model_config, plot_graph

accuracy_measures = {}

for activation_function in ["sigmoid", "tanh", "relu"]:

    model_config = base_model_config()
    model_config["HIDDEN_ACTIVATION"] = activation_function
    X, Y = get_data()
    model_name = f"Hidden-Activation-{activation_function}"
    history = create_and_run_model(
        model_config,
        X, 
        Y,
        model_name,
    )

    accuracy_measures[model_name] = history.history["accuracy"]

plot_graph(accuracy_measures, "Varying Hidden Layer Activation Functions")

