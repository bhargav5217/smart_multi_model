import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from modules.pipeline import build_preprocessor


def prepare_data(df, target):
    df = df.copy()
    df = df.drop_duplicates()

    X = df.drop(columns=[target])
    y = df[target]

    if y.dtype == "object":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True, max_iter=500)
    }


def train_models(df, target):
    X_train, X_test, y_train, y_test = prepare_data(df, target)
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(df, target)

    models = get_models()
    results = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = round(time.time() - start, 3)

        start = time.time()
        preds = pipeline.predict(X_test)
        infer_time = round(time.time() - start, 3)

        proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

        results[name] = {
            "pipeline": pipeline,
            "preds": preds,
            "proba": proba,
            "train_time": train_time,
            "infer_time": infer_time
        }

    return results, X_train, X_test, y_train, y_test, numeric_cols, categorical_cols


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
                    roc_auc_score(
                        y_test,
                        roc_input,
                        multi_class="ovr" if n_classes > 2 else "raise"
                    ),
                    3
                )
            except:
                roc = None

        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, preds), 3),
            "F1 Score": round(f1_score(y_test, preds, average=avg, zero_division=0), 3),
            "Precision": round(precision_score(y_test, preds, average=avg, zero_division=0), 3),
            "Recall": round(recall_score(y_test, preds, average=avg, zero_division=0), 3),
            "ROC-AUC": roc,
            "Train Time(s)": res["train_time"],
            "Infer Time(s)": res["infer_time"]
        })

    return pd.DataFrame(rows)


def rank_models(metrics_df, metric):
    ranked = metrics_df.sort_values(by=metric, ascending=False).reset_index(drop=True)
    return ranked


def plot_comparison(metrics_df, metric):
    df = metrics_df[["Model", metric]].dropna().sort_values(metric)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df["Model"], df[metric], color="steelblue")
    ax.set_title(f"Model Comparison - {metric}")
    ax.set_xlabel(metric)
    plt.tight_layout()
    return fig