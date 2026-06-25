import pandas as pd


def load_csv(file):
    df = pd.read_csv(file)
    return df


def basic_info(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum())
    }


def validate_target(df, target):
    if target not in df.columns:
        return False, "Selected target column is not in the dataset."

    if df[target].isnull().sum() > 0:
        return False, "Target column contains missing values. Please clean it first."

    if df[target].nunique() < 2:
        return False, "Target column must contain at least 2 classes."

    return True, "Target is valid."


def detect_problem_type(df, target):
    col = df[target]

    if col.dtype == "object":
        return "classification"

    if col.nunique() <= 15:
        return "classification"

    return "regression"