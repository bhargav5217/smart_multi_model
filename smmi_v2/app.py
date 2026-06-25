import streamlit as st
import pandas as pd

from modules.data import load_csv, basic_info, validate_target, detect_problem_type
from modules.eda import missing_summary, plot_missing, plot_target, plot_correlation
from modules.models import train_models, get_metrics, rank_models, plot_comparison
from modules.predict import build_query_input

st.set_page_config(page_title="SMMI V2", layout="wide")

st.title("Smart Multi-Model Implementer (V2)")
st.write("Upload a CSV dataset, run EDA, compare models, and test a single query prediction.")
st.divider()

# Step 1: Upload
st.subheader("Step 1: Upload Dataset")
file = st.file_uploader("Choose a CSV file", type=["csv"])

if file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = load_csv(file)
info = basic_info(df)

st.success("File loaded successfully.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", info["rows"])
col2.metric("Columns", info["columns"])
col3.metric("Missing Values", info["missing"])
col4.metric("Duplicates", info["duplicates"])

st.write("### Preview")
st.dataframe(df.head(), use_container_width=True)
st.divider()

# Step 2: Target column
st.subheader("Step 2: Select Target Column")
target = st.selectbox("Target column", df.columns.tolist())

valid, message = validate_target(df, target)
if not valid:
    st.error(message)
    st.stop()

problem_type = detect_problem_type(df, target)
st.info(f"Detected problem type: **{problem_type}**")

if problem_type != "classification":
    st.warning("Only classification is supported in V2.")
    st.stop()

st.write(f"Target classes: **{df[target].nunique()}**")
st.write(df[target].value_counts())

st.divider()

# Step 3: EDA
st.subheader("Step 3: Exploratory Data Analysis")

tab1, tab2, tab3 = st.tabs(["Missing Values", "Target Distribution", "Correlation"])

with tab1:
    missing_df = missing_summary(df)
    if missing_df.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(missing_df, use_container_width=True)
        fig = plot_missing(df)
        if fig:
            st.pyplot(fig)

with tab2:
    fig = plot_target(df, target)
    st.pyplot(fig)

with tab3:
    fig = plot_correlation(df)
    if fig:
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

st.divider()

# Step 4: Select ranking metric
st.subheader("Step 4: Choose Ranking Metric")
metric = st.selectbox(
    "Select metric to choose best model",
    ["F1 Score", "Accuracy", "Precision", "Recall", "ROC-AUC"]
)

st.divider()

# Step 5: Train models
st.subheader("Step 5: Train and Compare Models")

if st.button("Train Models"):
    with st.spinner("Training models..."):
        results, X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = train_models(df, target)
        metrics_df = get_metrics(results, y_test)
        ranked_df = rank_models(metrics_df, metric)

        st.session_state["results"] = results
        st.session_state["metrics_df"] = metrics_df
        st.session_state["ranked_df"] = ranked_df
        st.session_state["y_test"] = y_test
        st.session_state["target"] = target
        st.session_state["df"] = df

if "ranked_df" in st.session_state:
    st.success("All models trained successfully.")

    ranked_df = st.session_state["ranked_df"]
    results = st.session_state["results"]

    st.write("### Model Comparison Table")
    st.dataframe(ranked_df, use_container_width=True)

    st.write("### Comparison Chart")
    st.pyplot(plot_comparison(ranked_df, metric))

    st.divider()

    # Step 6: Best model
    st.subheader("Step 6: Best Model")

    best_row = ranked_df.dropna(subset=[metric]).iloc[0]
    best_model_name = best_row["Model"]
    best_pipeline = results[best_model_name]["pipeline"]

    st.success(f"Best model: **{best_model_name}** based on **{metric}**")

    st.write(f"- Accuracy: {best_row['Accuracy']}")
    st.write(f"- F1 Score: {best_row['F1 Score']}")
    st.write(f"- Precision: {best_row['Precision']}")
    st.write(f"- Recall: {best_row['Recall']}")
    st.write(f"- ROC-AUC: {best_row['ROC-AUC']}")
    st.write(f"- Train Time: {best_row['Train Time(s)']} sec")
    st.write(f"- Inference Time: {best_row['Infer Time(s)']} sec")

    st.divider()

    # Step 7: Query point prediction
    query_df = build_query_input(st.session_state["df"], st.session_state["target"])

    if st.button("Predict for Query Point"):
        pred = best_pipeline.predict(query_df)[0]
        st.success(f"Predicted class: **{pred}**")

        # if hasattr(best_pipeline, "predict_proba"):
        #     proba = best_pipeline.predict_proba(query_df)[0]
        #     st.write("Prediction probabilities:")
        #     st.write(proba)
        if hasattr(best_pipeline, "predict_proba"):
            proba = best_pipeline.predict_proba(query_df)[0]
            proba_df = pd.DataFrame({
                "Class": best_pipeline.classes_,
                "Probability": proba
            })
            st.write("Prediction probabilities:")
            st.dataframe(proba_df, use_container_width=True)