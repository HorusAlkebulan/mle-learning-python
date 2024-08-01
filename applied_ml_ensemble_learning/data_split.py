# data_split.py
import pandas as pd 
import os
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(__file__)
HR = "\n-----------------------------------------------------------------------"
RANDOM_SEED = 42

print(HR)
clean_data_path = os.path.join(PROJECT_ROOT, "titanic_cleaned.csv")
print(f"Loading cleaned data from {clean_data_path}")
titanic = pd.read_csv(clean_data_path)
print(f"Data loaded:\n{titanic.head(10)}")

features = titanic.drop("Survived", axis=1)
labels = titanic["Survived"]

# split into train + non_train
train_X, non_train_X, train_Y, non_train_Y = train_test_split(features, labels, test_size=0.4, random_state=RANDOM_SEED)

# split non_train into test and val
test_X, val_X, test_Y, val_Y = train_test_split(non_train_X, non_train_Y, test_size=0.5, random_state=RANDOM_SEED)

total_size = len(labels)

for dataset in [train_Y, test_Y, val_Y]:
    dataset_size = len(dataset)
    print(round(dataset_size/total_size, 2))

# saved split datasets
print(HR)

save_paths_dict = {
    "train_X": os.path.join(PROJECT_ROOT, "train_X.csv"),
    "test_X": os.path.join(PROJECT_ROOT, "test_X.csv"),    
    "val_X": os.path.join(PROJECT_ROOT, "val_X.csv"),
    "train_Y": os.path.join(PROJECT_ROOT, "train_Y.csv"),
    "test_Y": os.path.join(PROJECT_ROOT, "test_Y.csv"),
    "val_Y": os.path.join(PROJECT_ROOT, "val_Y.csv"),
}
print(f"Saving split data to:\n{save_paths_dict}")
train_X.to_csv(save_paths_dict["train_X"], index=False)
test_X.to_csv(save_paths_dict["test_X"], index=False)
val_X.to_csv(save_paths_dict["val_X"], index=False)
train_Y.to_csv(save_paths_dict["train_Y"], index=False)
test_Y.to_csv(save_paths_dict["test_Y"], index=False)
val_Y.to_csv(save_paths_dict["val_Y"], index=False)
