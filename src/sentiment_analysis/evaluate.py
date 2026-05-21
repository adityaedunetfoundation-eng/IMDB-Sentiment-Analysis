from sklearn.metrics import classification_report, accuracy_score


def evaluate_model(model, texts: list[str], labels: list[int]) -> str:
    """Evaluate the trained model and return a classification report."""
    predictions = model.predict(texts)
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=["negative", "positive"])
    return f"Accuracy: {accuracy:.4f}\n\n{report}"
