from pathlib import Path
import joblib, yaml
from sklearn.linear_model import LogisticRegression

with open("config/config.yaml") as f:
    CFG = yaml.safe_load(f)

def train_and_save(X_train, y_train):
    model = LogisticRegression(**CFG["model"]["params"])
    model.fit(X_train, y_train)
    out_dir = Path(CFG["paths"]["models"])
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
