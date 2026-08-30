# Heart Disease Streamlit Prototype

This app uses the best model produced by `Heart_Disease_Analysis.ipynb`:

`../models/notebook_review/best_model.joblib`

For the current notebook run, the selected model is Random Forest. The app sends
the same 20 input fields through the saved preprocessing and modelling pipeline,
then displays its positive-class probability and the classification at a fixed
0.50 threshold.

## Run

```powershell
cd <project-folder>\prototype
python -m pip install -r requirements.txt
.\run_app.ps1
```

Open the local address printed by Streamlit, normally `http://localhost:8501`.

## Files used

- `../models/notebook_review/best_model.joblib`
- `../models/notebook_review/model_metadata.json`

The paths are resolved from `model_adapter.py`, so they still work if the whole
project folder is moved. A different model can be tested with:

```powershell
$env:HEART_MODEL_PATH = '<path>\best_model.joblib'
$env:HEART_METADATA_PATH = '<path>\model_metadata.json'
python -m streamlit run .\app.py
```

## Interpretation

This is an assignment prototype rather than a clinical diagnostic system. The
three models produced ROC-AUC results close to 0.5, so the app mainly demonstrates
the complete data-preparation, model-loading and prediction workflow.
