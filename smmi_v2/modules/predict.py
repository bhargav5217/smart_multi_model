import pandas as pd
import streamlit as st


def build_query_input(df, target):
    feature_df = df.drop(columns=[target])
    input_data = {}

    st.subheader("Step 6: Query Point Prediction")
    st.write("Enter values for a single prediction:")

    for col in feature_df.columns:
        if feature_df[col].dtype == "object" or str(feature_df[col].dtype) == "category":
            options = feature_df[col].dropna().unique().tolist()
            if len(options) == 0:
                options = [""]
            input_data[col] = st.selectbox(f"{col}", options)
        elif str(feature_df[col].dtype) == "bool":
            input_data[col] = st.selectbox(f"{col}", [True, False])
        else:
            min_val = float(feature_df[col].min()) if feature_df[col].notnull().any() else 0.0
            max_val = float(feature_df[col].max()) if feature_df[col].notnull().any() else 100.0
            mean_val = float(feature_df[col].mean()) if feature_df[col].notnull().any() else 0.0

            input_data[col] = st.number_input(
                f"{col}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )

    query_df = pd.DataFrame([input_data])
    return query_df