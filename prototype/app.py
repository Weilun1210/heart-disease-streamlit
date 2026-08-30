"""Interactive heart-disease model demonstration for BMDS2003."""

from __future__ import annotations

import streamlit as st

from model_adapter import decision_threshold, load_artifacts, score_one


st.set_page_config(
    page_title="Heart Disease Risk Research Demo",
    page_icon="❤️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 1.6rem;}
      [data-testid="stMetricValue"] {font-size: 2.5rem;}
      .research-note {border-left: 5px solid #3D8DFF; background:#eef7ff; color:#0b1220 !important;
                      padding: .85rem 1rem; margin: .5rem 0 1.2rem 0;}
      div[data-testid="stForm"] {border: 1px solid #dce3ea; border-radius: 12px; padding: 1.1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Heart Disease Risk — Research Prototype")
st.caption("BMDS2003 Data Science Assignment")


@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_artifacts()


try:
    model, metadata, model_path = get_artifacts()
except Exception as exc:
    st.error(f"Model is not ready: {exc}")
    st.info(
        "Expected artifact: `models/notebook_review/best_model.joblib`. "
        "Run the modelling workflow first, then refresh this page."
    )
    st.stop()

exploratory_threshold = decision_threshold(metadata)
best_model_meta = metadata.get("best_model") if isinstance(metadata.get("best_model"), dict) else {}
model_name = str(
    best_model_meta.get("display_name")
    or best_model_meta.get("name")
    or metadata.get("model_name")
    or metadata.get("best_model_name")
    or metadata.get("algorithm")
    or type(model).__name__
)

with st.sidebar:
    st.header("Model information")
    st.write(f"**Selected model:** {model_name}")
    st.write(f"**Artifact:** `{model_path.name}`")
    threshold = exploratory_threshold
    st.write(f"**Active decision threshold:** {threshold:.3f}")

    test_metrics = best_model_meta.get("metrics_default_0_5") if isinstance(best_model_meta, dict) else None
    if isinstance(test_metrics, dict):
        st.write(
            f"**Held-out ROC-AUC:** {float(test_metrics.get('roc_auc', 0.0)):.3f}  \n"
            f"**Held-out PR-AUC:** {float(test_metrics.get('average_precision_pr_auc', 0.0)):.3f}"
        )
    st.divider()
    st.markdown(
        "**How to read the score**\n\n"
        "The probability is the model's estimated positive-class score. "
        f"A score of **{threshold:.1%} or above** is classified as the higher-score class."
    )
    st.divider()

st.subheader("Enter one observation")
st.caption("Defaults are close to the dataset medians or most common categories. Adjust any field, then select **Estimate score**.")

with st.form("risk_form", clear_on_submit=False):
    st.markdown("#### Demographics and lifestyle")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=49, step=1)
        gender = st.selectbox("Gender", ["Female", "Male"], index=1)
    with c2:
        exercise = st.selectbox("Exercise habits", ["Low", "Medium", "High"], index=2)
        smoking = st.selectbox("Smoking", ["No", "Yes"], index=1)
    with c3:
        alcohol = st.selectbox("Alcohol consumption", ["None", "Low", "Medium", "High"], index=2)
        stress = st.selectbox("Stress level", ["Low", "Medium", "High"], index=1)
    with c4:
        sleep = st.number_input("Sleep hours", min_value=0.0, max_value=16.0, value=7.0, step=0.1)
        sugar = st.selectbox("Sugar consumption", ["Low", "Medium", "High"], index=0)

    st.markdown("#### History and indicator flags")
    h1, h2, h3, h4, h5, h6 = st.columns(6)
    with h1:
        family = st.selectbox("Family heart disease", ["No", "Yes"], index=0)
    with h2:
        diabetes = st.selectbox("Diabetes", ["No", "Yes"], index=0)
    with h3:
        high_bp = st.selectbox("High blood pressure", ["No", "Yes"], index=1)
    with h4:
        low_hdl = st.selectbox("Low HDL cholesterol", ["No", "Yes"], index=1)
    with h5:
        high_ldl = st.selectbox("High LDL cholesterol", ["No", "Yes"], index=0)
    with h6:
        st.write("")
        st.caption("Flags use the dataset's Yes/No coding.")

    st.markdown("#### Clinical measurements")
    m1, m2, m3 = st.columns(3)
    with m1:
        bp = st.number_input("Blood pressure (mmHg)", min_value=70.0, max_value=250.0, value=150.0, step=1.0)
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=29.1, step=0.1)
        crp = st.number_input("CRP level", min_value=0.0, max_value=50.0, value=7.5, step=0.1)
    with m2:
        cholesterol = st.number_input("Cholesterol level (mg/dL)", min_value=50.0, max_value=500.0, value=226.0, step=1.0)
        triglyceride = st.number_input("Triglyceride level (mg/dL)", min_value=20.0, max_value=800.0, value=250.0, step=1.0)
    with m3:
        fasting = st.number_input("Fasting blood sugar (mg/dL)", min_value=40.0, max_value=400.0, value=120.0, step=1.0)
        homocysteine = st.number_input("Homocysteine level", min_value=0.0, max_value=60.0, value=12.4, step=0.1)

    submitted = st.form_submit_button("Estimate score", type="primary", width="stretch")

if submitted:
    values = {
        "Age": float(age),
        "Gender": gender,
        "Blood Pressure": float(bp),
        "Cholesterol Level": float(cholesterol),
        "Exercise Habits": exercise,
        "Smoking": smoking,
        "Family Heart Disease": family,
        "Diabetes": diabetes,
        "BMI": float(bmi),
        "High Blood Pressure": high_bp,
        "Low HDL Cholesterol": low_hdl,
        "High LDL Cholesterol": high_ldl,
        "Alcohol Consumption": alcohol,
        "Stress Level": stress,
        "Sleep Hours": float(sleep),
        "Sugar Consumption": sugar,
        "Triglyceride Level": float(triglyceride),
        "Fasting Blood Sugar": float(fasting),
        "CRP Level": float(crp),
        "Homocysteine Level": float(homocysteine),
    }
    try:
        probability, scored_row = score_one(model, values)
    except Exception as exc:
        st.error(f"Scoring failed: {exc}")
    else:
        higher = probability >= threshold
        label = "Higher model score" if higher else "Lower model score"
        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Positive-class probability", f"{probability:.1%}")
            if higher:
                st.error(f"Classification: {label}")
            else:
                st.success(f"Classification: {label}")
        with r2:
            st.markdown(
                f'<div class="research-note"><b>Threshold interpretation</b><br>'
                f"The score is {'above' if higher else 'below'} the configured {threshold:.1%} threshold. "
                "Because cross-validated ROC-AUC was approximately 0.5, the output demonstrates the pipeline rather than dependable clinical discrimination.</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Review the model input row"):
                display_row = scored_row.T.rename(columns={0: "Value"})
                display_row["Value"] = display_row["Value"].astype(str)
                st.dataframe(display_row, width="stretch")
