# ch05_fine_tune_training.py

from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForImageClassification,
    AutoFeatureExtractor,
)
# from ch03_defining_a_model import get_device, get_labels_and_mapping, MODEL_ID
# from ch04_preprocessing_images import (
#     IMAGE_KEY,
#     LABEL_KEY,
#     LABELS_KEY,
#     PIXEL_VALUES_KEY,
#     PROJECT_ROOT,
#     get_transformed_datasets,
# )

import os
from huggingface_hub import login
from torch.utils.data import DataLoader
import torch
import evaluate
import numpy as np
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
    Resize,
    CenterCrop,
)
import torch
import os
from transformers import (
    AutoModelForImageClassification,
    AutoFeatureExtractor,
)
from datasets import Dataset, load_from_disk, load_dataset

PROJECT_ROOT = os.path.dirname(__file__)
MODEL_ID = "google/vit-base-patch16-224"
PIXEL_VALUES_KEY = "pixel_values"
IMAGE_KEY = "image"
LABEL_KEY = "label"
LABELS_KEY = "labels"
BATCH_SIZE = 32
METRIC_ACCURACY = "accuracy"
FINE_TUNED_MODEL_NAME = "vit-base-patch16-224-finetuned-flower"
LOGGING_DIR = "logs"
TRAINING_BATCH_SIZE = 4

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

device = get_device()

def get_labels_and_mapping(PROJECT_ROOT):
    ds_train_path = os.path.join(PROJECT_ROOT, "data/ds_train.ds")
    print(f"Loading training set: {ds_train_path}")
    ds_train = load_from_disk(ds_train_path)

    labels = ds_train.features["label"].names
    print(f"labels: {labels}\n{type(labels)}")

    id2label = {key: value for key, value in enumerate(labels)}
    label2id = {value: key for key, value in enumerate(labels)}
    print(f"id2label: {id2label}")
    print(f"label2id: {label2id}")
    return labels,id2label,label2id

def get_huggingface_write_token() -> str:
    if "HUGGINGFACE_WRITE" in os.environ.keys():
        return os.environ["HUGGINGFACE_WRITE"]
    else:
        raise RuntimeError("Please set HUGGINGFACE_WRITE in environment variables.")




if __name__ == "__main__":

    logging_path = os.path.join(
        PROJECT_ROOT,
        LOGGING_DIR,
    )

    print(
        "Logging into huggingface (NOTE: This requires a huggingface WRITE token type)..."
    )
    login(
        token=get_huggingface_write_token(),
        add_to_git_credential=True,
    )

    # transformed_ds_train, transformed_ds_val, transformed_ds_test = get_transformed_datasets(PROJECT_ROOT, MODEL_ID, PIXEL_VALUES_KEY, IMAGE_KEY)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    normalize = Normalize(
        mean=feature_extractor.image_mean,
        std=feature_extractor.image_std,
    )

    size_feature_value = feature_extractor.size["height"]
    print(f"feature_extractor.size: {size_feature_value}")

    # setting up transform chains for train and validation
    train_transform = Compose(
        [
            RandomResizedCrop(size_feature_value),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    validation_transform = Compose(
        [
            Resize(size_feature_value),
            CenterCrop(size_feature_value),
            ToTensor(),
            normalize,
        ]
    )

    def collate_fn(images):
        labels = torch.tensor([image[LABEL_KEY] for image in images])
        pixel_values = torch.stack([image[PIXEL_VALUES_KEY] for image in images])
        return {
            PIXEL_VALUES_KEY: pixel_values,
            LABELS_KEY: labels,
        }

    def train_transform_images(images):
        images[PIXEL_VALUES_KEY] = [
            train_transform(image.convert("RGB")) for image in images[IMAGE_KEY]
        ]
        return images

    def validation_transform_images(images):
        images[PIXEL_VALUES_KEY] = [
            validation_transform(image.convert("RGB")) for image in images[IMAGE_KEY]
        ]

    # print(f"Loading datasets")
    # ds_train_path = os.path.join(PROJECT_ROOT, "data/ds_train.ds")
    # ds_val_path = os.path.join(PROJECT_ROOT, "data/ds_val.ds")
    # ds_test_path = os.path.join(PROJECT_ROOT, "data/ds_test.ds")

    # print(f"Loading datasets from {ds_train_path}, {ds_val_path}, and {ds_test_path}")
    # ds_train: Dataset = load_from_disk(ds_train_path)
    # ds_val: Dataset = load_from_disk(ds_val_path)
    # ds_test: Dataset = load_from_disk(ds_test_path)

    data_files = "https://github.com/jonfernandes/flowers-dataset/raw/main/flower_photos.tgz"
    local_path = "advanced_ai_transformers_cv/data/downloaded" # /flower_photos.tgz"
    # root_dir = "/Users/Horus/git-projects/horusalkebulan/mle-learning-python"
    # local_path = os.path.join(root_dir, local_path)
    local_path = "imagefolder"

    print(f"Loading (or downloading) dataset using {local_path} from URL {data_files}")
    ds = load_dataset(local_path, data_files=data_files)

    # data will land in the cache: /Users/Horus/.cache/huggingface/datasets/imagefolder/default-062daa9e8cf14967/0.0.0/37fbb85cc714a338bea574ac6c7d0b5be5aff46c1862c1989b20e0771199e93f/dataset_info.json
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

    ds_train: Dataset = ds["train"]
    ds_val: Dataset = ds["validation"]
    ds_test: Dataset = ds["test"]

    print(f"Transforming the datasets")
    transformed_ds_train = ds_train.with_transform(train_transform_images)
    transformed_ds_val = ds_val.with_transform(validation_transform_images)
    transformed_ds_test = ds_test.with_transform(validation_transform_images)

    print(f"transformed_ds_train: {transformed_ds_train}")
    print(f"transformed_ds_val: {transformed_ds_val}")
    print(f"transformed_ds_test: {transformed_ds_test}")

    train_rows = transformed_ds_train.num_rows
    val_rows = transformed_ds_val.num_rows
    test_rows = transformed_ds_test.num_rows
    total_rows = train_rows + val_rows + test_rows

    print(
        f"train/val/test split: {train_rows/total_rows:0.2f}/{val_rows/total_rows:0.2f}/{test_rows/total_rows:0.2f}"
    )

    # load up data for training, validation, and evaluation
    train_dataloader = DataLoader(
        transformed_ds_train,
        batch_size=TRAINING_BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True,
    )
    print(f"train_dataloader: {train_dataloader}")
    validation_dataloader = DataLoader(
        transformed_ds_val,
        batch_size=TRAINING_BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        transformed_ds_test,
        batch_size=TRAINING_BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # view batch
    batch_train = next(iter(train_dataloader))
    for key, value in batch_train.items():
        print(f"train_dataloader: key={key}, value={value.shape}")
    batch_val = next(iter(validation_dataloader))
    for key, value in batch_val.items():
        print(f"validation_dataloader: key={key}, value={value.shape}")
    batch_test = next(iter(test_dataloader))
    for key, value in batch_test.items():
        print(f"test_dataloader: key={key}, value={value.shape}")

    labels, id2label, label2id = get_labels_and_mapping(PROJECT_ROOT)

    device = get_device()

    print(
        f"Loading ViTransformer Model (for transfer learning) and Feature Extractor to {device}"
    )
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    print(f"model loaded: {model}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    if device == torch.device("mps"):
        print(f"MPS device found for training")
        use_mps_device = True
    else:
        use_mps_device = False

    args = TrainingArguments(
        FINE_TUNED_MODEL_NAME,
        evaluation_strategy="steps",
        learning_rate=5e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=1,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_ACCURACY,
        remove_unused_columns=False,
        logging_dir=logging_path,
        push_to_hub=False,
        use_mps_device=use_mps_device,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=transformed_ds_train,
        eval_dataset=transformed_ds_val,
        tokenizer=feature_extractor,
        data_collator=collate_fn,
        # optimizers=(torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR),
    )

    print(f"trainer loaded: {trainer}, running quick training test...")
    trainer.train()

    print("Predicion test using test data...")
    result = trainer.predict(transformed_ds_test)

    print(f"result: {result}")

    print("Setting up for full training run...")
    metric = evaluate.load(METRIC_ACCURACY)

    def compute_metrics_fn(batch):
        return metric.compute(
            references=batch.label_ids, predictions=np.argmax(batch.predictions, axis=1)
        )

    args = TrainingArguments(
        FINE_TUNED_MODEL_NAME,
        evaluation_strategy="steps",
        learning_rate=5e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_ACCURACY,
        remove_unused_columns=False,
        logging_dir=logging_path,
        push_to_hub=True,
        use_mps_device=use_mps_device,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=transformed_ds_train,
        eval_dataset=transformed_ds_val,
        tokenizer=feature_extractor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_fn,
        # optimizers=(torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR),
    )

    print(f"trainer loaded: {trainer}, running full training...")
    trainer.train()

    print("Saving trained model")
    trainer.save_model()

    print("Run evaluation using training data")
    trainer.evaluate(transformed_ds_train)

    print("Run evaluation using validation data")
    trainer.evaluate(transformed_ds_val)

    print("Run evalution using test data")
    trainer.evaluate(transformed_ds_test)
