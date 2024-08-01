# weight_initialization_experiment.py

from common_experiment_functions import get_data, create_and_run_model, base_model_config, plot_graph

accuracy_measures = {}

for initializer in ["random_normal", "random_uniform", "zeros", "ones"]:

    model_config = base_model_config()
    model_config["WEIGHTS_INITIALIZER"] = initializer
    X, Y = get_data()
    model_name = f"Weights-Initializer-{initializer}"
    history = create_and_run_model(
        model_config,
        X, 
        Y,
        model_name,
    )

    accuracy_measures[model_name] = history.history["accuracy"]

plot_graph(accuracy_measures, "Varying Weights Initializers")

