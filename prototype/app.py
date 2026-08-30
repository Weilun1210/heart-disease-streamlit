"""Interactive heart-disease data and model research prototype for BMDS2003."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from model_adapter import (
    decision_threshold,
    load_model_collection,
    project_root,
    score_model,
)


st.set_page_config(
    page_title="Heart Disease Research Prototype",
    page_icon=":heart:",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1220px; padding-top: 1.35rem; padding-bottom: 2.5rem;}
      [data-testid="stMetricValue"] {font-size: 2.2rem;}
      .research-note {
          border-left: 5px solid #3D8DFF;
          background: rgba(61,141,255,.10);
          padding: .9rem 1rem;
          border-radius: .25rem;
          margin: .5rem 0 1rem 0;
      }
      .small-note {opacity: .78; font-size: .92rem;}
      div[data-testid="stForm"] {border: 0; padding: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = project_root()
DATA_PATH = ROOT / "data" / "raw" / "heart_disease.csv"
OUTPUT_ROOT = ROOT / "prototype" / "model_outputs"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TARGET = "Heart Disease Status"


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def get_models():
    return load_model_collection()


@st.cache_data(show_spinner=False)
def load_model_results() -> pd.DataFrame | None:
    path = TABLE_DIR / "model_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_output_metadata() -> dict:
    candidates = [
        OUTPUT_ROOT / "model_outputs.json",
        ROOT / "models" / "notebook_review" / "model_metadata.json",
    ]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if isinstance(value, dict):
                return value
    return {}


def clean_display_name(name: str) -> str:
    return str(name).replace("_", " ").strip()


def dataset_default_numeric(df: pd.DataFrame, column: str, fallback: float) -> float:
    value = pd.to_numeric(df[column], errors="coerce").median()
    return float(value) if pd.notna(value) else float(fallback)


def dataset_default_category(df: pd.DataFrame, column: str, fallback: str) -> str:
    mode = df[column].dropna().mode()
    return str(mode.iloc[0]) if not mode.empty else fallback


def target_bar_figure(df: pd.DataFrame):
    counts = df[TARGET].value_counts().reindex(["No", "Yes"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Heart Disease Status")
    ax.set_xlabel("Status")
    ax.set_ylabel("Records")
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{int(value):,}", ha="center", va="bottom")
    fig.tight_layout()
    return fig


def missing_bar_figure(df: pd.DataFrame):
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0].head(10).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(missing.index.astype(str), missing.values)
    ax.set_title("Top Missing-Value Rates")
    ax.set_xlabel("Missing (%)")
    ax.set_ylabel("")
    for i, value in enumerate(missing.values):
        ax.text(value, i, f" {value:.2f}%", va="center")
    fig.tight_layout()
    return fig


def numeric_feature_figure(df: pd.DataFrame, feature: str):
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for label in ["No", "Yes"]:
        values = pd.to_numeric(
            df.loc[df[TARGET] == label, feature], errors="coerce"
        ).dropna()
        ax.hist(values, bins=28, alpha=0.55, density=True, label=label)
    ax.set_title(f"{clean_display_name(feature)} by Heart Disease Status")
    ax.set_xlabel(clean_display_name(feature))
    ax.set_ylabel("Density")
    ax.legend(title="Heart Disease")
    fig.tight_layout()
    return fig


def categorical_rate_figure(df: pd.DataFrame, feature: str):
    temp = df[[feature, TARGET]].copy()
    temp[feature] = temp[feature].fillna("Missing")
    temp["Positive"] = (temp[TARGET] == "Yes").astype(float)
    rate = temp.groupby(feature, dropna=False)["Positive"].mean().mul(100).sort_values()

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    ax.barh(rate.index.astype(str), rate.values)
    ax.set_title(f"Heart Disease Rate by {clean_display_name(feature)}")
    ax.set_xlabel("Positive class rate (%)")
    ax.set_ylabel("")
    for i, value in enumerate(rate.values):
        ax.text(value, i, f" {value:.1f}%", va="center")
    fig.tight_layout()
    return fig


def model_metric_figure(results: pd.DataFrame):
    metrics = ["CV ROC-AUC", "Test ROC-AUC", "PR-AUC"]
    available = [col for col in metrics if col in results.columns]
    plot_df = results[["Model", *available]].copy()

    x = np.arange(len(plot_df))
    width = 0.22 if len(available) == 3 else 0.3
    fig, ax = plt.subplots(figsize=(9, 4.8))
    offsets = np.linspace(-width, width, len(available)) if len(available) > 1 else [0]
    for offset, metric in zip(offsets, available):
        ax.bar(x + offset, plot_df[metric].astype(float), width=width, label=metric)
    ax.axhline(0.50, linestyle="--", linewidth=1, label="ROC-AUC = 0.50 reference")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Model"].astype(str))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Evaluation Summary")
    ax.legend()
    fig.tight_layout()
    return fig


def render_data_explorer(df: pd.DataFrame):
    st.header(":bar_chart: Data Explorer")
    st.caption("Explore the same heart-disease dataset used in the modelling notebook.")

    positive_rate = (df[TARGET] == "Yes").mean() * 100
    max_missing = df.isna().mean().max() * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records", f"{len(df):,}")
    m2.metric("Input features", f"{df.shape[1] - 1}")
    m3.metric("Positive class", f"{positive_rate:.1f}%")
    m4.metric("Highest missing rate", f"{max_missing:.2f}%")

    c1, c2 = st.columns(2)
    with c1:
        fig = target_bar_figure(df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("The target is imbalanced: most records are in the No class.")
    with c2:
        fig = missing_bar_figure(df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Alcohol Consumption contains substantially more missing values than the other variables.")

    st.divider()
    st.subheader(":mag: Explore one feature")

    numeric_features = [
        column for column in df.select_dtypes(include=np.number).columns if column != TARGET
    ]
    categorical_features = [
        column for column in df.select_dtypes(exclude=np.number).columns if column != TARGET
    ]

    tab1, tab2 = st.tabs(["Numerical feature", "Categorical feature"])
    with tab1:
        feature = st.selectbox("Choose a numerical feature", numeric_features, index=0)
        fig = numeric_feature_figure(df, feature)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        no_median = pd.to_numeric(df.loc[df[TARGET] == "No", feature], errors="coerce").median()
        yes_median = pd.to_numeric(df.loc[df[TARGET] == "Yes", feature], errors="coerce").median()
        st.caption(
            f"Median for No: {no_median:.2f}  |  Median for Yes: {yes_median:.2f}. "
            "The chart is exploratory and does not by itself establish predictive importance."
        )

    with tab2:
        feature = st.selectbox("Choose a categorical feature", categorical_features, index=0)
        fig = categorical_rate_figure(df, feature)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Bars show the observed positive-class percentage within each category in the dataset.")

    with st.expander(":page_facing_up: Preview dataset"):
        st.dataframe(df.head(30), use_container_width=True, hide_index=True)



def model_score_figure(result: dict, threshold: float):
    """Create a compact visual for the score returned by the selected model."""
    kind = result.get("score_kind")

    if kind == "positive-class probability":
        yes_score = float(result.get("display_score", 0.0))
        yes_score = min(max(yes_score, 0.0), 1.0)
        no_score = 1.0 - yes_score

        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        bars = ax.bar(["No", "Yes"], [no_score, yes_score])
        ax.axhline(float(threshold), linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Model probability")
        ax.set_title("No vs Yes Model Score")

        for bar, value in zip(bars, [no_score, yes_score]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.1%}",
                ha="center",
                va="bottom",
            )

        ax.text(
            1.02,
            float(threshold),
            f"Threshold {float(threshold):.0%}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=9,
        )
        fig.tight_layout()
        return fig

    if kind == "decision-function margin":
        margin = float(result.get("display_score", 0.0))
        limit = max(1.0, abs(margin) * 1.4)

        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        ax.barh(["SVM"], [margin])
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlim(-limit, limit)
        ax.set_xlabel("Decision-function margin")
        ax.set_title("SVM Model Score")
        ax.text(
            margin,
            0,
            f" {margin:.3f}" if margin >= 0 else f"{margin:.3f} ",
            va="center",
            ha="left" if margin >= 0 else "right",
        )
        fig.tight_layout()
        return fig

    return None

def render_model_score(df: pd.DataFrame, models: dict, paths: dict, metadata: dict):
    st.header(":test_tube: Model Score")
    st.caption("Enter a profile and view the output from one of the trained assignment models.")

    preferred_order = ["Random Forest", "Logistic Regression", "SVM"]
    available_names = [name for name in preferred_order if name in models] + [
        name for name in models if name not in preferred_order
    ]
    if not available_names:
        st.error("No saved model files are available yet.")
        return

    top_left, top_right = st.columns([2.2, 1])
    with top_left:
        selected_model = st.selectbox(
            "Choose model",
            available_names,
            index=available_names.index("Random Forest")
            if "Random Forest" in available_names
            else 0,
        )
    with top_right:
        st.markdown("**CV-selected model**")
        st.write("Random Forest")

    model = models[selected_model]
    threshold = decision_threshold(metadata)

    st.info(
        "Research prototype only. The notebook reported ROC-AUC values close to 0.50, "
        "so this demonstrates the trained pipeline rather than a dependable clinical diagnosis."
    )

    def n(column: str, fallback: float) -> float:
        return dataset_default_numeric(df, column, fallback)

    def c(column: str, fallback: str) -> str:
        return dataset_default_category(df, column, fallback)

    yes_no = ["No", "Yes"]

    with st.form("score_form", clear_on_submit=False):

        with st.container(border=True):
            st.markdown("### Demographics and lifestyle")
            st.caption("Basic profile and lifestyle information.")

            a1, a2, a3, a4 = st.columns(4)
            with a1:
                age = st.number_input(
                    "Age (years)", 18, 100, int(round(n("Age", 49))), 1
                )
                gender_opts = ["Female", "Male"]
                gender_default = c("Gender", "Male")
                gender = st.selectbox(
                    "Gender",
                    gender_opts,
                    index=gender_opts.index(gender_default)
                    if gender_default in gender_opts
                    else 0,
                )

            with a2:
                exercise_opts = ["Low", "Medium", "High"]
                exercise_default = c("Exercise Habits", "High")
                exercise = st.selectbox(
                    "Exercise habits",
                    exercise_opts,
                    index=exercise_opts.index(exercise_default)
                    if exercise_default in exercise_opts
                    else 0,
                )
                smoking_default = c("Smoking", "Yes")
                smoking = st.selectbox(
                    "Smoking",
                    yes_no,
                    index=yes_no.index(smoking_default)
                    if smoking_default in yes_no
                    else 0,
                )

            with a3:
                alcohol_opts = ["Not recorded", "Low", "Medium", "High"]
                alcohol = st.selectbox(
                    "Alcohol consumption", alcohol_opts, index=0
                )
                stress_opts = ["Low", "Medium", "High"]
                stress_default = c("Stress Level", "Medium")
                stress = st.selectbox(
                    "Stress level",
                    stress_opts,
                    index=stress_opts.index(stress_default)
                    if stress_default in stress_opts
                    else 0,
                )

            with a4:
                sleep = st.number_input(
                    "Sleep hours",
                    0.0,
                    16.0,
                    round(n("Sleep Hours", 7.0), 1),
                    0.1,
                )
                sugar_opts = ["Low", "Medium", "High"]
                sugar_default = c("Sugar Consumption", "Low")
                sugar = st.selectbox(
                    "Sugar consumption",
                    sugar_opts,
                    index=sugar_opts.index(sugar_default)
                    if sugar_default in sugar_opts
                    else 0,
                )

        st.write("")
        st.write("")

        with st.container(border=True):
            st.markdown("### History and indicator flags")
            st.caption("Existing history and binary health indicators.")

            b1, b2, b3 = st.columns(3)
            with b1:
                family_default = c("Family Heart Disease", "No")
                family = st.selectbox(
                    "Family heart disease",
                    yes_no,
                    index=yes_no.index(family_default)
                    if family_default in yes_no
                    else 0,
                )
            with b2:
                diabetes_default = c("Diabetes", "No")
                diabetes = st.selectbox(
                    "Diabetes",
                    yes_no,
                    index=yes_no.index(diabetes_default)
                    if diabetes_default in yes_no
                    else 0,
                )
            with b3:
                high_bp_default = c("High Blood Pressure", "Yes")
                high_bp = st.selectbox(
                    "High blood pressure",
                    yes_no,
                    index=yes_no.index(high_bp_default)
                    if high_bp_default in yes_no
                    else 0,
                )

            st.write("")

            b4, b5, b6 = st.columns(3)
            with b4:
                low_hdl_default = c("Low HDL Cholesterol", "Yes")
                low_hdl = st.selectbox(
                    "Low HDL cholesterol",
                    yes_no,
                    index=yes_no.index(low_hdl_default)
                    if low_hdl_default in yes_no
                    else 0,
                )
            with b5:
                high_ldl_default = c("High LDL Cholesterol", "No")
                high_ldl = st.selectbox(
                    "High LDL cholesterol",
                    yes_no,
                    index=yes_no.index(high_ldl_default)
                    if high_ldl_default in yes_no
                    else 0,
                )
            with b6:
                st.write("")

        st.write("")
        st.write("")

        with st.container(border=True):
            st.markdown("### Clinical measurements")
            st.caption("Numerical measurements used by the trained model pipeline.")

            d1, d2 = st.columns(2)
            with d1:
                bp = st.number_input(
                    "Blood pressure (mmHg)",
                    70.0,
                    250.0,
                    round(n("Blood Pressure", 150.0), 1),
                    1.0,
                )
                bmi = st.number_input(
                    "BMI (kg/m2)",
                    10.0,
                    70.0,
                    round(n("BMI", 29.1), 1),
                    0.1,
                )
                triglyceride = st.number_input(
                    "Triglyceride level (mg/dL)",
                    20.0,
                    800.0,
                    round(n("Triglyceride Level", 250.0), 1),
                    1.0,
                )
                crp = st.number_input(
                    "CRP level",
                    0.0,
                    50.0,
                    round(n("CRP Level", 7.5), 1),
                    0.1,
                )

            with d2:
                cholesterol = st.number_input(
                    "Cholesterol level (mg/dL)",
                    50.0,
                    500.0,
                    round(n("Cholesterol Level", 226.0), 1),
                    1.0,
                )
                fasting = st.number_input(
                    "Fasting blood sugar (mg/dL)",
                    40.0,
                    400.0,
                    round(n("Fasting Blood Sugar", 120.0), 1),
                    1.0,
                )
                homocysteine = st.number_input(
                    "Homocysteine level",
                    0.0,
                    60.0,
                    round(n("Homocysteine Level", 12.4), 1),
                    0.1,
                )

        st.write("")
        submitted = st.form_submit_button(
            "Calculate model score",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

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
        "Alcohol Consumption": np.nan if alcohol == "Not recorded" else alcohol,
        "Stress Level": stress,
        "Sleep Hours": float(sleep),
        "Sugar Consumption": sugar,
        "Triglyceride Level": float(triglyceride),
        "Fasting Blood Sugar": float(fasting),
        "CRP Level": float(crp),
        "Homocysteine Level": float(homocysteine),
    }

    try:
        result, scored_row = score_model(model, values)
    except Exception as exc:
        st.error(f"Scoring failed: {exc}")
        return

    st.markdown("### :pushpin: Model result")

    with st.container(border=True):
        r1, r2, r3 = st.columns(3)
        r1.metric("Model", selected_model)

        if result["score_kind"] == "positive-class probability":
            score = float(result["display_score"])
            r2.metric("Positive-class score", f"{score:.1%}")
            r3.metric("Predicted class", result["prediction"])
        elif result["score_kind"] == "decision-function margin":
            margin = float(result["display_score"])
            r2.metric("SVM decision margin", f"{margin:.3f}")
            r3.metric("Predicted class", result["prediction"])
        else:
            r2.metric("Model score", "N/A")
            r3.metric("Predicted class", result["prediction"])

        fig = model_score_figure(result, threshold)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if result["score_kind"] == "positive-class probability":
            st.caption(
                f"The dashed line marks the default {threshold:.0%} decision threshold."
            )
        elif result["score_kind"] == "decision-function margin":
            st.caption(
                "For SVM, zero is the decision boundary. This margin is not a calibrated probability."
            )

    st.caption(
        "Assignment demonstration only. The notebook evaluation showed weak discrimination, "
        "so this output should not be used for medical diagnosis or treatment decisions."
    )

    with st.expander(":page_facing_up: Review exact model input"):
        display_row = scored_row.T.rename(columns={0: "Value"})
        display_row["Value"] = display_row["Value"].where(
            display_row["Value"].notna(), "Missing"
        )
        st.dataframe(display_row, use_container_width=True)

def render_model_evaluation(results: pd.DataFrame | None, metadata: dict):
    st.header(":chart_with_upwards_trend: Model Evaluation")
    st.caption("Results exported from the modelling notebook. The app does not retrain the models.")

    if results is None:
        st.warning(
            "The exported model-results files are not present yet. Run the updated notebook from top to bottom, then refresh this app."
        )
        best = metadata.get("best_model") if isinstance(metadata.get("best_model"), dict) else {}
        metrics = best.get("metrics_default_0_5") if isinstance(best, dict) else {}
        if metrics:
            st.subheader("Saved best-model metadata")
            c1, c2, c3 = st.columns(3)
            c1.metric("CV ROC-AUC", f"{float(best.get('cv_roc_auc', 0)):.3f}")
            c2.metric("Test ROC-AUC", f"{float(metrics.get('roc_auc', 0)):.3f}")
            c3.metric("PR-AUC", f"{float(metrics.get('average_precision_pr_auc', 0)):.3f}")
        return

    display = results.copy()
    numeric_cols = [c for c in display.columns if c != "Model"]
    for col in numeric_cols:
        display[col] = pd.to_numeric(display[col], errors="coerce")

    best_cv = display.sort_values("CV ROC-AUC", ascending=False).iloc[0]
    best_test = display.sort_values("Test ROC-AUC", ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CV-selected model", str(best_cv["Model"]))
    c2.metric("Best CV ROC-AUC", f"{best_cv['CV ROC-AUC']:.3f}")
    c3.metric("Highest Test ROC-AUC", f"{best_test['Test ROC-AUC']:.3f}")
    c4.metric("Positive prevalence", f"{float(metadata.get('positive_class_rate', 0.20)):.1%}")

    st.markdown(
        '<div class="research-note"><b>Selection rule:</b> the recommended model is chosen using the highest 5-fold training CV ROC-AUC. '
        "The held-out test set is reported afterward for comparison.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Model comparison")
    st.dataframe(display.round(3), use_container_width=True, hide_index=True)
    fig = model_metric_figure(display)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader(":dart: Why accuracy alone is misleading")
    baseline = metadata.get("majority_class_reference") if isinstance(metadata, dict) else None
    if isinstance(baseline, dict):
        b1, b2, b3 = st.columns(3)
        b1.metric("Always-No accuracy", f"{float(baseline.get('accuracy', 0.8)):.1%}")
        b2.metric("Always-No recall", f"{float(baseline.get('recall', 0.0)):.1%}")
        b3.metric("Always-No ROC-AUC", f"{float(baseline.get('roc_auc', 0.5)):.3f}")
    st.write(
        "Because 80% of the dataset belongs to the No class, a classifier that predicts No for every record already achieves 80% accuracy while detecting none of the positive cases."
    )

    st.divider()
    st.subheader("Notebook evaluation figures")
    figure_specs = [
        ("model_comparison.png", "Model comparison"),
        ("confusion_matrices.png", "Confusion matrices"),
        ("roc_curves.png", "ROC curves"),
        ("precision_recall_curves.png", "Precision-recall curves"),
    ]
    available = [(FIGURE_DIR / filename, caption) for filename, caption in figure_specs if (FIGURE_DIR / filename).exists()]

    if not available:
        st.info("The notebook figure exports will appear here after the updated notebook has been run.")
    else:
        for path, caption in available:
            st.image(str(path), caption=caption, use_container_width=True)

    st.divider()
    st.subheader("Main finding")
    st.write(
        "The three formal models produced ROC-AUC values close to 0.50. Random Forest was still selected correctly under the predefined CV rule, but the small scores indicate that the supplied variables provide weak predictive discrimination for this dataset."
    )


# ---------- app bootstrap ----------
st.title(":heart: Heart Disease Data & Model Research Prototype")
st.caption("BMDS2003 Data Science Assignment")

try:
    data = load_dataset()
except Exception as exc:
    st.error(f"Dataset is not ready: {exc}")
    st.stop()

try:
    models, model_paths, model_metadata = get_models()
except Exception as exc:
    models, model_paths, model_metadata = {}, {}, load_output_metadata()
    model_error = str(exc)
else:
    model_error = None

output_metadata = load_output_metadata()
if output_metadata:
    model_metadata = {**model_metadata, **output_metadata}

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        [":bar_chart: Data Explorer", ":test_tube: Model Score", ":chart_with_upwards_trend: Model Evaluation"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("This prototype is for assignment demonstration and research use only.")
    if model_error:
        st.warning("Model artifacts are not fully available yet.")

if page == ":bar_chart: Data Explorer":
    render_data_explorer(data)
elif page == ":test_tube: Model Score":
    if not models:
        st.error(f"Model artifacts are not ready: {model_error}")
        st.info("Run the updated modelling notebook first, then refresh this page.")
    else:
        render_model_score(data, models, model_paths, model_metadata)
else:
    render_model_evaluation(load_model_results(), model_metadata)
