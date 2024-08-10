# defining_a_model.py
import os
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch
from datasets import Dataset, load_from_disk
from PIL.JpegImagePlugin import JpegImageFile

PROJECT_ROOT = os.path.dirname(__file__)
MODEL_ID = "google/vit-base-patch16-224"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


ds_train_path = os.path.join(PROJECT_ROOT, "data/ds_train.ds")
print(f"Loading training set: {ds_train_path}")
ds_train = load_from_disk(ds_train_path)

labels = ds_train.features["label"].names
print(f"labels: {labels}\n{type(labels)}")

id2label = {key: value for key, value in enumerate(labels)}
label2id = {value: key for key, value in enumerate(labels)}
print(f"id2label: {id2label}")
print(f"label2id: {label2id}")

print("Loading ViTransformer Model (for transfer learning) and Feature Extractor")
model = AutoModelForImageClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
model.to(device)
model.eval()

print(f"model loaded: {model}")
