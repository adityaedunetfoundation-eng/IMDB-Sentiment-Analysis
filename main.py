import argparse
from sentiment_analysis.train import train_model, save_model
from sentiment_analysis.evaluate import evaluate_model
from sentiment_analysis.data import load_imdb_dataset, prepare_data
from sentiment_analysis.model import load_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="IMDb sentiment analysis CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a sentiment classifier")
    train_parser.add_argument("--output-model", default="model.joblib", help="Output path for the trained model")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", default="model.joblib", help="Path to the trained model file")

    predict_parser = subparsers.add_parser("predict", help="Predict sentiment for a review")
    predict_parser.add_argument("--model", default="model.joblib", help="Path to the trained model file")
    predict_parser.add_argument("--review", required=True, help="Text review to classify")

    args = parser.parse_args()

    if args.command == "train":
        dataset = load_imdb_dataset()
        X_train, y_train = prepare_data(dataset["train"])
        model = train_model(X_train, y_train)
        save_model(model, args.output_model)
        print(f"Model trained and saved to {args.output_model}")

    elif args.command == "evaluate":
        dataset = load_imdb_dataset()
        X_test, y_test = prepare_data(dataset["test"])
        model = load_pipeline(args.model)
        report = evaluate_model(model, X_test, y_test)
        print(report)

    elif args.command == "predict":
        model = load_pipeline(args.model)
        prediction = model.predict([args.review])[0]
        label = "positive" if prediction == 1 else "negative"
        print(label)


if __name__ == "__main__":
    main()
