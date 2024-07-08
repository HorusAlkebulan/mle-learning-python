# data_loading.py
import torch
from torchvision.datasets import CIFAR10
from keras.datasets import cifar10
from matplotlib import pyplot
import logging

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
# logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_log_handler = logging.FileHandler(f"{__file__}.log")
file_log_handler.setFormatter(formatter)
logger.addHandler(file_log_handler)

console_log_handler = logging.StreamHandler()
console_log_handler.setFormatter(formatter)
logger.addHandler(console_log_handler)

# import dataset
logger.info(f"Downloading CIFAR10 data to ./train")

train_data = CIFAR10(root="./train/",
                     train=True,
                     download=True)

logger.info(f"Initalized train_data={train_data}")
