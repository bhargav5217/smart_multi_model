import streamlit as st
import pandas as pd

from modules.data import (
    load_csv, basic_info, detect_problem_type,
    missing_summary, plot_missing, plot_target, plot_correlation
)
from modules.models import (
    preprocess, get_model_list, train_single_model,
    get_metrics, plot_comparison, plot_feature_importance
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="SMMI v2", layout="wide", page_icon="🤖")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 SMMI v2")
    st.markdown("**Smart Multi-Model Implementer**")
    st.divider()
    st.markdown("### Steps")
    st.markdown("1. Upload CSV\n2. Pick target\n3. Run EDA\n4. Train models\n5. Compare & explain")
    st.divider()
    st.caption("Built with scikit-learn + Streamlit")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("Smart Multi-Model Implementer")
st.write("Upload any CSV → auto EDA → train 6 classifiers → compare performance.")
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Step 1 · Upload Dataset")
file = st.file_uploader("Choose a CSV file", type=["csv"])

if file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = load_csv(file)
info = basic_info(df)

st.success(f"✅ File loaded: **{file.name}**")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows",          info["rows"])
c2.metric("Columns",       info["columns"])
c3.metric("Missing Vals",  info["missing"])
c4.metric("Duplicates",    info["duplicates"])

with st.expander("Preview data (first 5 rows)"):
    st.dataframe(df.head(), use_container_width=True)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Target column
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Step 2 · Select Target Column")
target = st.selectbox("Target column:", df.columns.tolist())

problem_type = detect_problem_type(df, target)
st.info(f"Detected problem type: **{problem_type}**")

if problem_type == "regression":
    st.warning("⚠️ Regression support coming in V3. Please use a classification dataset.")
    st.stop()

st.write(f"Unique classes: **{df[target].nunique()}**")
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — EDA (cached)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Step 3 · Exploratory Data Analysis")

@st.cache_data
def run_eda(data_bytes, tgt):
    import io
    d = pd.read_csv(io.BytesIO(data_bytes))
    return (
        missing_summary(d),
        plot_missing(d),
        plot_target(d, tgt),
        plot_correlation(d),
    )

file.seek(0)
raw_bytes = file.read()

with st.spinner("Running EDA..."):
    miss_df, fig_miss, fig_target, fig_corr = run_eda(raw_bytes, target)

tab1, tab2, tab3 = st.tabs(["Missing Values", "Target Distribution", "Correlation"])

with tab1:
    if miss_df.empty:
        st.success("No missing values — clean dataset!")
    else:
        st.dataframe(miss_df, use_container_width=True)
        if fig_miss:
            st.pyplot(fig_miss)

with tab2:
    st.pyplot(fig_target)

with tab3:
    if fig_corr:
        st.pyplot(fig_corr)
    else:
        st.write("Not enough numeric columns for correlation heatmap.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Train models (with per-model progress)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Step 4 · Train All Models")
st.caption("Models: Logistic Regression · Decision Tree · Random Forest · Gradient Boosting · SVM · KNN")

if st.button("🚀 Train Models", type="primary"):
    with st.spinner("Preprocessing..."):
        X_train, X_test, y_train, y_test = preprocess(df, target)

    st.write(f"Train: `{X_train.shape}` | Test: `{X_test.shape}`")

    model_list = get_model_list()
    results = {}
    progress = st.progress(0)
    status   = st.empty()

    for i, (name, model) in enumerate(model_list.items()):
        status.text(f"Training {name}...")
        results[name] = train_single_model(name, model, X_train, y_train, X_test)
        progress.progress((i + 1) / len(model_list))

    status.empty()
    progress.empty()

    metrics_df = get_metrics(results, y_test)

    st.session_state["metrics_df"]  = metrics_df
    st.session_state["results"]     = results
    st.session_state["X_train"]     = X_train
    st.session_state["trained"]     = True

if not st.session_state.get("trained"):
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Compare
# ═══════════════════════════════════════════════════════════════════════════════
metrics_df = st.session_state["metrics_df"]

st.success("✅ All models trained!")
st.divider()
st.subheader("Step 5 · Compare All Models")
st.dataframe(metrics_df, use_container_width=True)

metric = st.selectbox("Rank models by:", ["F1 Score", "Accuracy", "ROC-AUC", "Precision", "Recall"])
st.pyplot(plot_comparison(metrics_df, metric))
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Best model + Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Step 6 · Best Model & Feature Importance")

best_row  = metrics_df.dropna(subset=[metric]).iloc[0]
best_name = best_row["Model"]

st.success(f"🏆 Best model: **{best_name}** (by {metric} = **{best_row[metric]}**)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy",     best_row["Accuracy"])
col2.metric("F1 Score",     best_row["F1 Score"])
col3.metric("ROC-AUC",      best_row["ROC-AUC"] if best_row["ROC-AUC"] else "N/A")
col4.metric("Train Time",   f"{best_row['Train Time(s)']}s")

# Feature importance for tree-based models
best_model   = st.session_state["results"][best_name]["model"]
feature_cols = st.session_state["X_train"].columns.tolist()
fig_fi = plot_feature_importance(best_model, feature_cols, best_name)

if fig_fi:
    st.pyplot(fig_fi)
else:
    st.info(f"ℹ️ Feature importance chart is available for tree-based models. "
            f"{best_name} does not support it directly.")

st.divider()
st.caption("SMMI v2 · Smart Multi-Model Implementer · Built with scikit-learn + Streamlit")
