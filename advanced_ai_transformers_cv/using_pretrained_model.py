# using_pretrained_model.py

from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch
from datasets import Dataset, load_from_disk
from PIL.JpegImagePlugin import JpegImageFile
import os

PROJECT_ROOT = os.path.dirname(__file__)

model_id = "google/vit-base-patch16-224"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Loading AutoModelForImageClassification pretrained as '{model_id}' to device '{device}'")
model = AutoModelForImageClassification.from_pretrained(model_id).to(device)
model.eval()

print(f"model: {model}")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
print(f"feature_extractor: {feature_extractor}")

print(f"Loading datasets")
ds_train_path = os.path.join(PROJECT_ROOT, "data/ds_train.ds")
ds_val_path = os.path.join(PROJECT_ROOT, "data/ds_val.ds")
ds_test_path = os.path.join(PROJECT_ROOT, "data/ds_test.ds")
print(f"Loading datasets from {ds_train_path}, {ds_val_path}, and {ds_test_path}")

ds_train: Dataset = load_from_disk(ds_train_path)
ds_val: Dataset = load_from_disk(ds_val_path)
ds_test: Dataset = load_from_disk(ds_test_path)

sample_image_id = 3
sample_image: JpegImageFile = ds_train[sample_image_id]['image']
sample_image_path = os.path.join(PROJECT_ROOT, f"sample_image_{sample_image_id}.png")
print(f"Saving sample image as {sample_image_path}")
sample_image.save(sample_image_path)

train_image_batch_feature = feature_extractor(
    images=sample_image,
    return_tensors="pt",
).to(device)
print(f"train_image_batch_feature: {train_image_batch_feature}")
train_image_t = torch.tensor(train_image_batch_feature["pixel_values"]).to(device)
print(f"train_image_t: {train_image_t}")
output_t = model(**train_image_batch_feature)
print(f"output_t: {output_t}")
print(f"output_t.logits: {type(output_t.logits)} {output_t.logits.shape} {output_t.logits}")
output_max = torch.argmax(output_t.logits, dim=1)
print(f"output_max: {output_max}")
output_int = output_max.item()
print(f"output_int: {output_int}")
print(f"model.config: {model.config}")
output_label = model.config.id2label[output_int]
print(f"output_label for {output_int}: {output_label}")
