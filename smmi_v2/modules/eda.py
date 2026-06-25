import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def missing_summary(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        return pd.DataFrame()

    percent = (missing / len(df) * 100).round(2)

    return pd.DataFrame({
        "Missing Count": missing,
        "Missing %": percent
    }).sort_values("Missing %", ascending=False)


def plot_missing(df):
    missing = (df.isnull().sum() / len(df) * 100)
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    missing.plot(kind="bar", color="tomato", ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_ylabel("Missing %")
    ax.set_xlabel("Columns")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_target(df, target):
    fig, ax = plt.subplots(figsize=(7, 4))
    df[target].value_counts().plot(kind="bar", color="steelblue", ax=ax)
    ax.set_title("Target Distribution")
    ax.set_xlabel(target)
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_correlation(df):
    num_df = df.select_dtypes(include=["int64", "float64"])

    if num_df.shape[1] < 2:
        return None

    # keep it light
    if num_df.shape[1] > 12:
        num_df = num_df.iloc[:, :12]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    return fig