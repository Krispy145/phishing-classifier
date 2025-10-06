# Minimal end-to-end runner
import argparse, joblib
from src.data.load import load_raw
from src.data.preprocess import split_xy, make_splits
from src.models.train import train_and_save
from src.models.evaluate import evaluate
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all","data","train","eval"], default="all")
    args = parser.parse_args()

    X_train = X_test = y_train = y_test = None
    df = load_raw()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = make_splits(X, y)

    if args.stage in ("all","train"):
        train_and_save(X_train, y_train)

    if args.stage in ("all","eval"):
        model_path = Path("app/models/model.joblib")
        if not model_path.exists():
            raise SystemExit("No model found. Run with --stage train first.")
        model = joblib.load(model_path)
        metrics = evaluate(model, X_test, y_test)
        print(metrics)

if __name__ == "__main__":
    main()
