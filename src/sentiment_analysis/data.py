from typing import Tuple

from datasets import DatasetDict, load_dataset


def load_imdb_dataset() -> DatasetDict:
    """Load the IMDb dataset from Hugging Face datasets."""
    return load_dataset("imdb")


def prepare_data(subset) -> Tuple[list[str], list[int]]:
    """Convert a dataset subset into text and label lists."""
    texts = [item["text"] for item in subset]
    labels = [int(item["label"]) for item in subset]
    return texts, labels
