# ch06_inference.py

import os
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from datasets import load_dataset

TEST_DS_KEY = "test"
VALIDATION_DS_KEY = "validation"
TRAIN_DS_KEY = "train"
PROJECT_ROOT = os.path.dirname(__file__)
IMAGE_KEY = "image"
LABEL_KEY = "label"
LABELS_KEY = "labels"
FINE_TUNED_MODEL_NAME = "vit-base-patch16-224-finetuned-flower"

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def classify_image(image):

    device = get_device()
    model = AutoModelForImageClassification.from_pretrained(FINE_TUNED_MODEL_NAME).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(FINE_TUNED_MODEL_NAME)
    input = feature_extractor(image, return_tensors="pt").to(device)
    output = model(**input)
    prediction = torch.argmax(output.logits, dim=-1).item()
    return model.config.id2label[prediction]

def classify_image_confidence(image):

    device = get_device()
    model = AutoModelForImageClassification.from_pretrained(FINE_TUNED_MODEL_NAME).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(FINE_TUNED_MODEL_NAME)
    input = feature_extractor(image, return_tensors="pt").to(device)
    output = model(**input)
    prediction = torch.nn.functional.softmax(output.logits, dim=-1)
    prediction_np = prediction[0].cpu().detach().numpy()
    confidence = {label: float(prediction_np[i]) for i, label in enumerate(labels)}
    return confidence


if __name__ == "__main__":

    device = get_device()

    data_files = (
        "https://github.com/jonfernandes/flowers-dataset/raw/main/flower_photos.tgz"
    )
    local_path = "advanced_ai_transformers_cv/data/downloaded"  # /flower_photos.tgz"
    # root_dir = "/Users/Horus/git-projects/horusalkebulan/mle-learning-python"
    # local_path = os.path.join(root_dir, local_path)
    local_path = "imagefolder"

    print(f"Loading (or downloading) dataset using {local_path} from URL {data_files}")
    ds = load_dataset(local_path, data_files=data_files)
    print(f"ds: {ds}")

    labels = ds["train"].features["label"].names

    ds_train_validation = ds["train"].train_test_split(
        test_size=0.1, seed=1, shuffle=True
    )
    ds_train_validation["validation"] = ds_train_validation.pop("test")
    ds.update(ds_train_validation)
    ds_train_test = ds["train"].train_test_split(test_size=0.1, seed=1, shuffle=True)
    ds.update(ds_train_test)

    test_image = ds[TEST_DS_KEY][-1][IMAGE_KEY]
    id2label = {key: value for key, value in enumerate(labels)}
    label2id = {value: key for key, value in enumerate(labels)}

    sample_image_path = os.path.join(PROJECT_ROOT, f"inference_original_test_image.jpg")
    print(f"Saving original sample training image: {sample_image_path}")
    test_image.save(sample_image_path)


    print(f"classify_image(test_image={test_image})")
    result = classify_image(test_image)
    print(f"result={result}")

    print(f"classify_image_confidence(test_image={test_image})")
    confidence = classify_image_confidence(test_image)
    print(f"confidence={confidence}")

