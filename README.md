# Smart Multi-Model Implementer

Upload any CSV, pick a target column, and the system automatically runs EDA,
trains 6 models, compares them, and tells you which one performs best.

## Run

```
pip install -r requirements.txt
streamlit run app.py
```

## Structure

smmi v1/

# Main Streamlit app
- app.py  

# Project dependencies
- requirements.txt  

# Python modules
- modules/  
  # Data loading, EDA, plots
  - data.py  
  
  # Preprocessing, training, evaluation
  - models.py  


