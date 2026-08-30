# Heart Disease Streamlit Prototype

This Streamlit app is the deployment prototype for `Heart_Disease_Analysis.ipynb`.
It uses the saved preprocessing + model pipelines created by the notebook and does
not retrain the models inside Streamlit.

## App pages

- **Data Explorer** - explores the same `heart_disease.csv` dataset used by the notebook.
- **Model Score** - scores one entered profile with Random Forest, Logistic Regression,
  or SVM when the corresponding saved model file is available.
- **Model Evaluation** - displays the evaluation tables and figures exported by the
  notebook.

## Model selection

The notebook compares three formal models using 5-fold stratified cross-validation
ROC-AUC on the training data. In the current run, **Random Forest is the CV-selected
model** because it has the highest training CV ROC-AUC among the three candidates.
The held-out test metrics are shown afterward for comparison.

The reported ROC-AUC values are close to 0.50, so this prototype should be treated as
an assignment/research demonstration rather than a clinical diagnostic system.

## Run

From the project folder:

```powershell
cd prototype
python -m pip install -r requirements.txt
.\run_app.ps1
```

Or run Streamlit directly:

```powershell
python -m streamlit run app.py
```

Open the local address printed by Streamlit, normally `http://localhost:8501`.

## Files used by the app

Dataset:

- `../data/raw/heart_disease.csv`

Saved model artifacts:

- `../models/notebook_review/random_forest.joblib`
- `../models/notebook_review/logistic_regression.joblib`
- `../models/notebook_review/svm.joblib`
- `../models/notebook_review/best_model.joblib` (fallback / selected-model copy)
- `../models/notebook_review/model_metadata.json`

Notebook-exported evaluation files:

- `model_outputs/model_outputs.json`
- `model_outputs/tables/model_results.csv`
- `model_outputs/tables/best_parameters.csv`
- `model_outputs/tables/majority_class_reference.csv`
- `model_outputs/tables/split_summary.csv`
- `model_outputs/tables/classification_reports.csv`
- `model_outputs/figures/model_comparison.png`
- `model_outputs/figures/confusion_matrices.png`
- `model_outputs/figures/roc_curves.png`
- `model_outputs/figures/precision_recall_curves.png`

## Scoring behaviour

Random Forest and Logistic Regression expose a positive-class probability through
`predict_proba()`.

The saved SVM exposes a `decision_function()` score rather than a calibrated
probability. The app therefore labels the SVM result as a **decision-function margin**
instead of presenting it as a probability.

`Alcohol Consumption` includes a **Not recorded** option. This is passed as a missing
value so the same imputation step learned in the notebook pipeline handles it.

## Recreating the exported evaluation outputs

Run `Heart_Disease_Analysis.ipynb` from top to bottom. The final export section writes
the Streamlit evaluation files into `prototype/model_outputs/` without retraining or
changing the already selected models.
