from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from joblib import load


def build_pipeline() -> Pipeline:
    """Build a scikit-learn pipeline for text classification."""
    return Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    strip_accents="unicode",
                    lowercase=True,
                    stop_words="english",
                    max_features=20000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )


def load_pipeline(path: str) -> BaseEstimator:
    """Load a persisted pipeline from disk."""
    return load(path)
