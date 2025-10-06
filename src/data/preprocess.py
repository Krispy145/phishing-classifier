import yaml
from sklearn.model_selection import train_test_split

with open("config/config.yaml") as f:
    CFG = yaml.safe_load(f)

LABEL_COL = "Result"  # adjust after inspecting CSV

def split_xy(df):
    y = df[LABEL_COL].values
    X = df.drop(columns=[LABEL_COL]).values
    return X, y

def make_splits(X, y):
    strat = y if CFG["split"]["stratify"] else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CFG["split"]["test_size"], stratify=strat, random_state=CFG["seed"]
    )
    return X_train, X_test, y_train, y_test
