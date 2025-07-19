import kagglehub
import os

os.environ["KAGGLEHUB_CACHE"] = "dataset"


def download_dataset(handle: str) -> None:
    """Downloads a dataset from Kaggle.

    Args:
        handle: The Kaggle dataset handle
    """
    try:
        local_dir = kagglehub.dataset_download(handle)
        print(f"Downloaded dataset {handle} to {local_dir}")
    except ValueError:
        local_dir = kagglehub.competition_download(handle)
        print(f"Downloaded dataset {handle} to {local_dir}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")


# download competition dataset
download_dataset("eedi-mining-misconceptions-in-mathematics")

# download datasets for fine-tuning
handles = [
    "conjuring92/eedi-five-folds",
    "conjuring92/eedi-silver-v3",
    "conjuring92/eedi-embed-pretrain-mix-final",
    "conjuring92/eedi-embed-mix-silver-v3",
    "conjuring92/eedi-ranker-silver-v3-teacher-blended-cot",
    "conjuring92/eedi-tutor-mix-v8",
    "conjuring92/eedi-cot-sonnet-6k",
    "conjuring92/eedi-cot-train-silver-v3",
    "conjuring92/eedi-misconception-clusters",
    "conjuring92/eedi-cot-gen-base",
]

for handle in handles:
    download_dataset(handle)
