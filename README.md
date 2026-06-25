---
title: SMMI V2
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Smart Multi-Model Implementer (SMMI) v2

Upload any CSV dataset → automatic EDA → preprocessing → train 6 models → compare performance → feature importance.

## What's New in V2
- Lightweight & fast — no freezing
- All steps cached with @st.cache_data
- Progress bar per model during training
- Feature importance chart for best model
- Clean sidebar navigation
- HF-ready README built in

## How to Use
1. Upload any CSV file
2. Select the target column
3. Run EDA, then Train Models
4. Compare results and see best model
