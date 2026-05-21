from sentiment_analysis.model import build_pipeline
from sentiment_analysis.evaluate import evaluate_model


def test_pipeline_predicts_on_simple_examples() -> None:
    texts = [
        "I loved this movie, it was amazing and fun.",
        "This film was terrible and boring.",
    ]
    labels = [1, 0]

    model = build_pipeline()
    model.fit(texts, labels)

    predictions = model.predict(texts)
    assert list(predictions) == labels

    report = evaluate_model(model, texts, labels)
    assert "Accuracy:" in report
