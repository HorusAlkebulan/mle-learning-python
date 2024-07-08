# data_loading.py
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from keras.datasets import cifar10
from matplotlib import pyplot
import logging
from torch.utils.data import DataLoader

"""
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

(echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv)"') >> /Users/horus/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"

brew install wget

Response stopped
Sent by Copilot:
The error you’re encountering while importing the CIFAR-10 dataset from keras.datasets seems to be related to SSL certificate verification. Here are a few steps you can try to resolve it:

Download the Dataset Manually:
Download the CIFAR-10 dataset from this link: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

mkdir data
wget -O ./data/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

Save it to a local directory on your machine.

mkdir -p ~/.keras/datasets

Rename the Downloaded File:
Rename the downloaded file to cifar-10-batches-py.tar.gz.

cp ./data/cifar-10-python.tar.gz ~/.keras/datasets
mv ~/.keras/datasets/cifar-10-python.tar.gz ~/.keras/datasets/cifar-10-batches-py.tar.gz

Copy to Keras Datasets Directory:
Copy the renamed file to the ~/.keras/datasets/ directory (create the datasets folder if it doesn’t exist).
After following these steps, try importing the dataset again using from keras.datasets import cifar10. This should resolve the issue. If you encounter any further problems, feel free to ask!
"""

logger = logging.getLogger(__name__)


def initialize_logger():
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_log_handler = logging.FileHandler(f"{__file__}.log")
    file_log_handler.setFormatter(formatter)
    logger.addHandler(file_log_handler)

    console_log_handler = logging.StreamHandler()
    console_log_handler.setFormatter(formatter)
    logger.addHandler(console_log_handler)


if __name__ == "__main__":

    initialize_logger()

    logger.info(f"Downloading CIFAR10 data to ./train")

    train_data = CIFAR10(root="./train/", train=True, download=True)

    logger.info(f"Initalized train_data={train_data}")

    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    logger.info("Dataset Summary: Shapes")
    logger.info(f"Train: X={train_x.shape}, Y={train_y.shape}")
    logger.info(f"Test:  X={test_x.shape}, Y={test_y.shape}")

    logger.info("Dataset Summary: Sample Data")
    logger.info(f"Train X[0]={train_x[0]}")
    logger.info(f"Train Y[0]={train_y[0]}")

    images_to_display = 16
    logger.info(f"Displaying {images_to_display} images from Training")
    nrows = 4
    ncols = 4
    for i in range(images_to_display):
        index = i + 1
        pyplot.subplot(nrows, ncols, index)
        pyplot.imshow(train_x[i])

    output_image_path = "data_loading.png"
    logger.info(f"Saving image as {output_image_path}")
    pyplot.savefig(output_image_path)

    logger.info(f"Training Data Shape: {train_data.data.shape}")
    logger.info(f"Labels: {train_data.class_to_idx}")

    logger.info("Training data with mapped labels")
    for i in range(images_to_display):
        data, label = train_data[i]
        logger.info(f"\tItem: {i}, Label: {label}, Class: {train_data.classes[label]}")

    train_data_path = "./train/"
    logger.info(f"Import and transform for training data set: {train_data_path}")
    transform_mean = (0.4914, 0.4822, 0.4465)
    transform_std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transform_mean,
                std=transform_std,
            ),
        ]
    )
    training_data = CIFAR10(
        train_data_path,
        train=True,
        download=True,
        transform=train_transform,
    )
    logger.info(f"Sample transformed training data: {training_data[0]}")

    test_data_path = "./test/"
    logger.info(f"Import and transform for testing data set: {test_data_path}")
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transform_mean,
                std=transform_std,
            ),
        ]
    )
    test_data = CIFAR10(
        test_data_path,
        train=False,
        download=True,
        transform=test_transform,
    )
    logger.info(f"Sample transformed test data: {test_data[0]}")

    batch_size = 16
    training_data_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )
    test_data_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )
    