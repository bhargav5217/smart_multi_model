import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)


def preprocess(df, target):
    df = df.copy()
    df = df.drop_duplicates()

    # Drop columns with >70% missing
    for col in df.columns:
        if col != target and df[col].isnull().mean() > 0.7:
            df = df.drop(columns=[col])

    X = df.drop(columns=[target])
    y = df[target]

    # Encode target if string
    if y.dtype == "object":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target)

    # OHE only — simple and reliable
    X = pd.get_dummies(X, drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Impute then scale
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test  = pd.DataFrame(imputer.transform(X_test),      columns=X_test.columns)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

    return X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def get_model_list():
    return {
        "Logistic Regression": LogisticRegression(max_iter=500, solver="lbfgs"),
        "Decision Tree":       DecisionTreeClassifier(max_depth=10),
        "Random Forest":       RandomForestClassifier(n_estimators=50, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=50),
        "SVM":                 SVC(probability=True, max_iter=500),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
    }


def train_single_model(name, model, X_train, y_train, X_test):
    """Train one model and return results dict."""
    start = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start, 3)

    start = time.time()
    preds = model.predict(X_test)
    infer_time = round(time.time() - start, 4)

    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    return {
        "model":      model,
        "preds":      preds,
        "proba":      proba,
        "train_time": train_time,
        "infer_time": infer_time,
    }


def get_metrics(results, y_test):
    rows = []
    n_classes = len(set(y_test))
    avg = "binary" if n_classes == 2 else "weighted"

    for name, res in results.items():
        preds = res["preds"]
        proba = res["proba"]

        roc = None
        if proba is not None:
            try:
                roc_input = proba[:, 1] if n_classes == 2 else proba
                roc = round(
                    roc_auc_score(y_test, roc_input,
                                  multi_class="ovr" if n_classes > 2 else "raise"), 3
                )
            except Exception:
                roc = None

        rows.append({
            "Model":        name,
            "Accuracy":     round(accuracy_score(y_test, preds), 3),
            "F1 Score":     round(f1_score(y_test, preds, average=avg, zero_division=0), 3),
            "Precision":    round(precision_score(y_test, preds, average=avg, zero_division=0), 3),
            "Recall":       round(recall_score(y_test, preds, average=avg, zero_division=0), 3),
            "ROC-AUC":      roc,
            "Train Time(s)":  res["train_time"],
            "Infer Time(s)":  res["infer_time"],
        })

    return pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)


def plot_comparison(metrics_df, metric):
    df = metrics_df[["Model", metric]].dropna().sort_values(metric)
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(df["Model"], df[metric], color="steelblue")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_title(f"Model Comparison — {metric}")
    ax.set_xlabel(metric)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, model_name):
    """Works for tree-based models. Returns None for others."""
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1], color="teal")
    ax.set_title(f"Top 15 Feature Importances — {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig
