from typing import Tuple
from joblib import dump

from sentiment_analysis.model import build_pipeline


def train_model(texts: list[str], labels: list[int]) -> object:
    """Train a sentiment classifier pipeline."""
    pipeline = build_pipeline()
    pipeline.fit(texts, labels)
    return pipeline


def save_model(model: object, output_path: str) -> None:
    """Persist the trained model to disk."""
    dump(model, output_path)
