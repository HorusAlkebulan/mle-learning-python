# model_lifecycle.py
from datetime import datetime
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
# from keras.datasets import cifar10
from matplotlib import pyplot
import logging
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models

# from torch import optim
import torch.optim as optim
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# NOTE: Training runtime on MAC CPU: about 7 minutes

def initialize_logger():
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_log_handler = logging.FileHandler(f"{__file__}.log")
    file_log_handler.setFormatter(formatter)
    logger.addHandler(file_log_handler)

    console_log_handler = logging.StreamHandler()
    console_log_handler.setFormatter(formatter)
    logger.addHandler(console_log_handler)


class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # wire up model
        self.num_classes = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=self.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
if __name__ == "__main__":

    initialize_logger()


    logger.info("Initializing the model")
    model = Net()
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=0.001,
        momentum=0.9
    )
    logger.info(f"Model initialized: {model}")
    logger.info(f"Model parameters: {model.parameters()}")

    training_loop_data_path = "./data"
    logger.info(f"Reloading data for training loop from: {training_loop_data_path}")
    transform_mean = (0.5, 0.5, 0.5)
    transform_std = (0.5, 0.5, 0.5)
    training_loop_tranform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(transform_mean, transform_std),
    ])
    training_data = CIFAR10(
        root=training_loop_data_path,
        train=True,
        transform=training_loop_tranform,
        download=True,
    )
    batch_size = 4
    num_workers = 2
    training_data_loader = DataLoader(
        dataset=training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_data = CIFAR10(
        root=training_loop_data_path,
        train=False,
        transform=training_loop_tranform,
        download=True,
    )
    test_data_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    logger.info(f"Training the network, starting: {datetime.now()}")

    epochs_to_train = 10
    report_stats_every = 2000

    logger.info(f"epochs_to_train: {epochs_to_train}")
    for epoch in range(epochs_to_train):
        logger.info(f"\tepoch: {epoch}")
        running_loss = 0.0
        for i, data in enumerate(training_data_loader, start=0):

            # get the input/label tuple
            inputs, labels = data

            # zero out gradients to prevent carry over
            optimizer.zero_grad()

            # forward, loss, backward, optimize round
            out = model(inputs)
            loss = loss_criterion(out, labels)
            loss.backward()
            optimizer.step()

            # log statistics
            running_loss = running_loss + loss.item()
            if i % report_stats_every == 0:
                logger.info(f"\t\t[batch: {i:5d}] loss: {running_loss / report_stats_every:.3f}")
                running_loss = 0.0

    logger.info(f"Training complete: {datetime.now()}")

    model_save_path = f"{__file__}.pt"
    logger.info(f"Saving model as {model_save_path}")
    torch.save(model, model_save_path)
