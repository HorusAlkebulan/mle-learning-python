# See README.md for setup

from datasets import load_dataset, Dataset
import os
from PIL.JpegImagePlugin import JpegImageFile
from IPython.display import display
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

data_files = "https://github.com/jonfernandes/flowers-dataset/raw/main/flower_photos.tgz"
local_path = "advanced_ai_transformers_cv/data/downloaded" # /flower_photos.tgz"
root_dir = "/Users/Horus/git-projects/horusalkebulan/mle-learning-python"
local_path = os.path.join(root_dir, local_path)
local_path = "imagefolder"

print(f"Loading (or downloading) dataset using {local_path} from URL {data_files}")
ds = load_dataset(local_path, data_files=data_files)

# data will land in the cache: /Users/Horus/.cache/huggingface/datasets/imagefolder/default-062daa9e8cf14967/0.0.0/37fbb85cc714a338bea574ac6c7d0b5be5aff46c1862c1989b20e0771199e93f/dataset_info.json

print("Viewing data")
for i in range(5):
    print("-----")
    print(f"Train image {i}")
    print(f"Type: {type(ds['train'][i]['image'])}")
    train_img: JpegImageFile = ds['train'][i]['image']
    save_image_path = f"train_image_{i}.png"
    print(f"Saved to '{save_image_path}' sized '{train_img.size}'")
    train_img.save(save_image_path)

labels = ds['train'].features['label'].names

print(f"Labels in the data: {labels}")

# splitting the data
ds_train_validation = ds["train"].train_test_split(
    test_size=0.1,
    seed=1,
    shuffle=True
)

print(f"ds_train_validation: {ds_train_validation}")

ds_train_validation["validation"] = ds_train_validation.pop("test")
print(f"ds_train_validation: {ds_train_validation}")

ds.update(ds_train_validation)
print(f"ds: {ds}")

ds_train_test = ds["train"].train_test_split(
    test_size=0.1,
    seed=1,
    shuffle=True
)
print(f"ds_train_test: {ds_train_test}")

ds.update(ds_train_test)
print(f"ds: {ds}")

train_rows = ds["train"].num_rows
val_rows = ds["validation"].num_rows
test_rows = ds["test"].num_rows
total_rows = train_rows + val_rows + test_rows

print(f"train/val/test split: {train_rows/total_rows:0.2f}/{val_rows/total_rows:0.2f}/{test_rows/total_rows:0.2f}")

ds_train_path = "data/ds_train.ds"
ds_train: Dataset = ds["train"]

ds_val_path = "data/ds_val.ds"
ds_val: Dataset = ds["validation"]

ds_test_path = "data/ds_test.ds"
ds_test: Dataset = ds["test"]

print(f"Persisting data to {ds_train_path}, {ds_val_path}, and {ds_test_path}")
ds_train.save_to_disk(ds_train_path)
ds_val.save_to_disk(ds_val_path)
ds_test.save_to_disk(ds_test_path)
