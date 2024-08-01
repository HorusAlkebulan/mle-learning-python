from common_experiment_functions import base_model_config, create_and_run_model, get_data, plot_graph


def batch_size_epoch_experiment():
    accuracy_measures = {}

    for batch_size in range(16, 128, 16):
        # load config using defaults
        model_config = base_model_config()

        # get and process data
        X, Y = get_data()

        model_config["EPOCHS"] = 20
        model_config["BATCH_SIZE"] = batch_size
        print(f"model_config: {model_config}")

        model_name = "Epochs-20-Batch-Size-" + str(batch_size)

        print(f"Creating and running a model '{model_name}")
        history = create_and_run_model(
            model_config,
            X,
            Y,
            model_name,
        )

        accuracy_measures[model_name] = history.history["accuracy"]

    plot_graph(accuracy_measures, "Compare Batch Size and Epoch")

if __name__ == "__main__":
    batch_size_epoch_experiment()