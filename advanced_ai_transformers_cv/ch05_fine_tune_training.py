# ch05_fine_tune_training.py

from transformers import TrainingArguments, Trainer
from ch04_preprocessing_images import PROJECT_ROOT
import os
from huggingface_hub import notebook_login, login
from torch.utils.data import DataLoader

BATCH_SIZE = 32
METRIC_ACCURACY = "accuracy"
FINE_TUNED_MODEL_NAME = "vit-base-patch16-224-finetuned-flower"
LOGGING_DIR = "logs"

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
    print("Logged in successfully.")

    

