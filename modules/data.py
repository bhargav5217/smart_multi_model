import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_csv(file):
    return pd.read_csv(file)

def basic_info(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum())
    }

def detect_problem_type(df, target):
    col = df[target]
    if col.dtype == "object":
        return "classification"
    if col.nunique() <= 15:
        return "classification"
    return "regression"

def missing_summary(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    percent = (missing / len(df) * 100).round(1)
    return pd.DataFrame({"Missing Count": missing, "Missing %": percent})

def plot_missing(df):
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 3))
    missing.plot(kind="bar", ax=ax, color="tomato")
    ax.set_title("Missing Values %")
    ax.set_ylabel("%")
    plt.tight_layout()
    return fig

def plot_target(df, target):
    fig, ax = plt.subplots(figsize=(6, 3))
    df[target].value_counts().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Target Distribution")
    ax.set_xlabel(target)
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig

def plot_correlation(df):
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return None
    # Limit to 15 columns max to avoid huge heatmaps freezing
    if num_df.shape[1] > 15:
        num_df = num_df.iloc[:, :15]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Correlation Heatmap (top 15 numeric cols)")
    plt.tight_layout()
    return fig
