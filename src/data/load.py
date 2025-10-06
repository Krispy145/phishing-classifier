from pathlib import Path
import pandas as pd
import yaml

with open("config/config.yaml") as f:
    CFG = yaml.safe_load(f)

def load_raw() -> pd.DataFrame:
    csv_path = Path(CFG["paths"]["raw"]) / CFG["dataset"]["filename"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected raw dataset at {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Loaded dataset is empty")
    return df
