import torchvision
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
    Resize,
    CenterCrop
)
import torch
import os
from transformers import (
    AutoModelForImageClassification,
    AutoFeatureExtractor,
)
from datasets import Dataset, load_from_disk
from PIL.JpegImagePlugin import JpegImageFile

PROJECT_ROOT = os.path.dirname(__file__)
MODEL_ID = "google/vit-base-patch16-224"
PIXEL_VALUES_KEY = "pixel_values"
IMAGE_KEY = "image"
LABEL_KEY = "label"
LABELS_KEY = "labels"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def get_transformed_datasets(PROJECT_ROOT, MODEL_ID, PIXEL_VALUES_KEY, IMAGE_KEY):
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    normalize = Normalize(
        mean=feature_extractor.image_mean,
        std=feature_extractor.image_std,
    )

    size_feature_value = feature_extractor.size['height']
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

    def train_transform_images(images):
        images[PIXEL_VALUES_KEY] = [
            train_transform(image.convert("RGB")) for image in images[IMAGE_KEY]
        ]
        return images
    
    def validation_transform_images(images):
        images[PIXEL_VALUES_KEY] = [
            validation_transform(image.convert("RGB")) for image in images[IMAGE_KEY]
        ]

    print(f"Loading datasets")
    ds_train_path = os.path.join(PROJECT_ROOT, "data/ds_train.ds")
    ds_val_path = os.path.join(PROJECT_ROOT, "data/ds_val.ds")
    ds_test_path = os.path.join(PROJECT_ROOT, "data/ds_test.ds")
    
    print(f"Loading datasets from {ds_train_path}, {ds_val_path}, and {ds_test_path}")
    ds_train: Dataset = load_from_disk(ds_train_path)
    ds_val: Dataset = load_from_disk(ds_val_path)
    ds_test: Dataset = load_from_disk(ds_test_path)

    print(f"Transforming the datasets")
    transformed_ds_train = ds_train.with_transform(train_transform_images)
    transformed_ds_val = ds_val.with_transform(validation_transform_images)
    transformed_ds_test = ds_test.with_transform(validation_transform_images)
    return transformed_ds_train, transformed_ds_val, transformed_ds_test

if __name__ == "__main__":

    transformed_ds_train, transformed_ds_val, transformed_ds_test = get_transformed_datasets(PROJECT_ROOT, MODEL_ID, PIXEL_VALUES_KEY, IMAGE_KEY)

    # view sample image
    for sample_image_id in range(5):
        sample_image: JpegImageFile = transformed_ds_train[sample_image_id][IMAGE_KEY]
        sample_image_path = os.path.join(PROJECT_ROOT, f"transformed_sample_image_{sample_image_id}.jpg")
        print(f"Saving transformed sample training image: {sample_image_path}")
        sample_image.save(sample_image_path)

    # next get the images in the correct format for batch processing
    image_batch = [transformed_ds_train[i] for i in range(4)]
    print(f"image_batch: {image_batch}")
    for i in range(4):
        print(f"Shape[{i}]: {image_batch[i][PIXEL_VALUES_KEY].shape}")
        print(f"Label[{i}]: {image_batch[i][LABEL_KEY]}")
    image_batch_labels_t = torch.tensor(
        [image[LABEL_KEY] for image in image_batch]
    )
    print(f"image_batch_labels_t: {image_batch_labels_t}")

    image_batch_pixel_values_t = torch.stack(
        [image[PIXEL_VALUES_KEY] for image in image_batch]
    )
    print(f"image_batch_pixel_values_t: {image_batch_pixel_values_t}\n{image_batch_pixel_values_t.shape}")

    # save the transformed datasets
    # NOTE: Doesn't work, get error
    #     TypeError: Object of type function is not JSON serializable
    # The format kwargs must be JSON serializable, but key 'transform' isn't.
    # REASON: I believe it's because we've applied transforms.

    # ds_train_path = os.path.join(PROJECT_ROOT, "data/transformed_ds_train.ds")
    # ds_val_path = os.path.join(PROJECT_ROOT, "data/transformed_ds_val.ds")
    # ds_test_path = os.path.join(PROJECT_ROOT, "data/transformed_ds_test.ds")
    
    # print(f"Saving transformed datasets to {ds_train_path}, {ds_val_path}, and {ds_test_path}")
    # transformed_ds_train.save_to_disk(ds_train_path)
    # transformed_ds_val.save_to_disk(ds_val_path)
    # transformed_ds_test.save_to_disk(ds_test_path)

    # INSTEAD: Load dynamically from transform script