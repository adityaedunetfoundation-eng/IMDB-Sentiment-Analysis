# IMDb Sentiment Analysis

This project builds a sentiment analysis system for IMDb movie reviews. It trains a classifier to predict whether a review is positive or negative using the IMDb Movie Reviews dataset.

## Features

- Loads IMDb reviews using the Hugging Face `datasets` library
- Trains a `scikit-learn` pipeline with TF-IDF and logistic regression
- Evaluates accuracy and classification metrics on the test split
- Provides a CLI for training, evaluation, and prediction

## Setup

1. Create a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

Train a model:

```powershell
python main.py train --output-model model.joblib
```

Evaluate the saved model:

```powershell
python main.py evaluate --model model.joblib
```

Predict a single review:

```powershell
python main.py predict --model model.joblib --review "This movie was amazing and emotionally powerful."
```
