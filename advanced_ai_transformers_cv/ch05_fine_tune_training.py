# ch05_fine_tune_training.py

from transformers import TrainingArguments, Trainer, AutoModelForImageClassification, AutoFeatureExtractor
from ch03_defining_a_model import get_device, get_labels_and_mapping, MODEL_ID
from ch04_preprocessing_images import IMAGE_KEY, LABEL_KEY, LABELS_KEY, PIXEL_VALUES_KEY, PROJECT_ROOT, get_transformed_datasets
import os
from huggingface_hub import login
from torch.utils.data import DataLoader
import torch

BATCH_SIZE = 32
METRIC_ACCURACY = "accuracy"
FINE_TUNED_MODEL_NAME = "vit-base-patch16-224-finetuned-flower"
LOGGING_DIR = "logs"
TRAINING_BATCH_SIZE = 4

def get_huggingface_write_token() -> str:
    if "HUGGINGFACE_WRITE" in os.environ.keys():
        return os.environ["HUGGINGFACE_WRITE"]
    else:
        raise RuntimeError("Please set HUGGINGFACE_WRITE in environment variables.")

def collate_fn(images):
    labels = torch.tensor(
        [image[LABEL_KEY] for image in images]
    )
    pixel_values = torch.stack(
        [image[PIXEL_VALUES_KEY] for image in images]
    )
    return {
        PIXEL_VALUES_KEY: pixel_values,
        LABELS_KEY: labels,
    }

if __name__ == "__main__":

    logging_path = os.path.join(
        PROJECT_ROOT,
        LOGGING_DIR,
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
        push_to_hub=True
    )

    print("Logging into huggingface (NOTE: This requires a huggingface WRITE token type)...")
    login(
        token=get_huggingface_write_token(),
        add_to_git_credential=True,
        )

    transformed_ds_train, transformed_ds_val, transformed_ds_test = get_transformed_datasets(PROJECT_ROOT, MODEL_ID, PIXEL_VALUES_KEY, IMAGE_KEY)

    # load up data for training, validation, and evaluation
    train_dataloader = DataLoader(
        transformed_ds_train,
        batch_size = TRAINING_BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True,
    )
    print(f"train_dataloader: {train_dataloader}")
    validation_dataloader = DataLoader(
        transformed_ds_val,
        batch_size=TRAINING_BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False
    )
    test_dataloader = DataLoader(
        transformed_ds_test,
        batch_size=TRAINING_BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # view batch
    batch = next(iter(train_dataloader))
    for key, value in batch.items():
        print(f"train_dataloader: key={key}, value={value.shape}")

    labels, id2label, label2id = get_labels_and_mapping(PROJECT_ROOT)

    device = get_device()

    print(f"Loading ViTransformer Model (for transfer learning) and Feature Extractor to {device}")
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
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=transformed_ds_train,
        eval_dataset=transformed_ds_val,
        tokenizer=feature_extractor,
        data_collator=collate_fn,
    )

    print(f"trainer loaded: {trainer}")