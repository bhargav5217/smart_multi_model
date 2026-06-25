import streamlit as st
import pandas as pd

from modules.data import (load_csv, basic_info, detect_problem_type,
                           missing_summary, plot_missing, plot_target, plot_correlation)
from modules.models import preprocess, train_models, get_metrics, plot_comparison

st.set_page_config(page_title="Smart Multi-Model Implementer", layout="wide")

st.title("Smart Multi-Model Implementer")
st.write("Upload a CSV file, pick your target column, and the system will train and compare all models.")
st.divider()

# Step 1: Upload
st.subheader("Step 1: Upload your dataset")
file = st.file_uploader("Choose a CSV file", type=["csv"])

if file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = load_csv(file)
info = basic_info(df)

st.success("File loaded!")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", info["rows"])
col2.metric("Columns", info["columns"])
col3.metric("Missing Values", info["missing"])
col4.metric("Duplicates", info["duplicates"])

st.write("**Preview:**")
st.dataframe(df.head())
st.divider()

# Step 2: Target column
st.subheader("Step 2: Select the target column")
target = st.selectbox("Target column:", df.columns.tolist())
st.write(f"Unique values: {df[target].nunique()}")
st.write(df[target].value_counts())

problem_type = detect_problem_type(df, target)
st.info(f"Detected type classification/regression: **{problem_type}**")

if problem_type == "regression":
    st.warning("Regression models are coming soon. Please use a classification dataset.")
    st.stop()

st.divider()

# Step 3: EDA
st.subheader("Step 3: Exploratory Data Analysis")

tab1, tab2, tab3 = st.tabs(["Missing Values", "Target Distribution", "Correlation"])

with tab1:
    missing = missing_summary(df)
    if missing.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(missing)
        fig = plot_missing(df)
        if fig:
            st.pyplot(fig)

with tab2:
    st.pyplot(plot_target(df, target))

with tab3:
    fig = plot_correlation(df)
    if fig:
        st.pyplot(fig)
    else:
        st.write("Not enough numeric columns for correlation.")

st.divider()

# Step 4: Train
st.subheader("Step 4: Train all models")
st.write("Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN")

if st.button("Train Models"):
    with st.spinner("Preprocessing data..."):
        X_train, X_test, y_train, y_test = preprocess(df, target)
        st.write(f"Train size: {X_train.shape} | Test size: {X_test.shape}")

    with st.spinner("Training models..."):
        results = train_models(X_train, y_train, X_test)
        metrics_df = get_metrics(results, y_test)

    st.session_state["metrics_df"] = metrics_df
    st.session_state["trained"] = True

if st.session_state.get("trained"):
    metrics_df = st.session_state["metrics_df"]

    st.success("All models trained!")
    st.divider()

    # Step 5: Compare
    st.subheader("Step 5: Compare all models")
    st.dataframe(metrics_df, use_container_width=True)

    metric = st.selectbox("Rank models by:", ["F1 Score", "Accuracy", "ROC-AUC", "Precision", "Recall"])
    st.pyplot(plot_comparison(metrics_df, metric))
    st.divider()

    # Step 6: Best model
    st.subheader("Step 6: Best model")
    best_row = metrics_df.dropna(subset=[metric]).iloc[0]
    best_name = best_row["Model"]

    st.success(f"Best model: **{best_name}** (based on {metric} = {best_row[metric]})")
    st.write(f"- Accuracy: {best_row['Accuracy']}")
    st.write(f"- F1 Score: {best_row['F1 Score']}")
    st.write(f"- ROC-AUC: {best_row['ROC-AUC']}")
    st.write(f"- Train time: {best_row['Train Time(s)']}s | Infer time: {best_row['Infer Time(s)']}s")
