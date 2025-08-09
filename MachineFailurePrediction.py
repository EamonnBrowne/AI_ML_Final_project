# app.py

###############################################################################
# 1. Throttle native BLAS/OpenMP and related thread pools (MUST be before any imports)
import os
import sys
os.environ["OMP_NUM_THREADS"]         = "1"   # OpenMP
os.environ["OPENBLAS_NUM_THREADS"]   = "1"   # OpenBLAS
os.environ["MKL_NUM_THREADS"]        = "1"   # Intel MKL
os.environ["NUMEXPR_NUM_THREADS"]    = "1"   # NumExpr
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # macOS-only (Accelerate)

###############################################################################
# 2. Suppress warnings & imports
import warnings
import json
import gc
from threadpoolctl import threadpool_limits
from streamlit.runtime.media_file_storage import MediaFileStorageError
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import shap
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.inspection import permutation_importance
from PIL import Image
import plotly.express as px

def cleanup_session_state():
    for key in list(st.session_state.keys()):
        if key.startswith("shap") or key in [
            "explainer", "background", "X_proc", "df_in", "exp",
            "shap_df", "fig_shap", "pi", "imp_df", "fig_imp",
            "X_array", "df_metrics", "avg", "prediction_result"
        ]:
            st.session_state.pop(key, None)

def cleanup_globals():
    mod = sys.modules[__name__]
    # don't delete these core modules, variables and utility functions)
    protected_vars = {
        "feature_names","updated_feature_names","target_names","X_test","y_test",
        "failure_descriptions","demo_input","thresholds",
        "page", "st", "pd", "np", "gc", "sys", "os", "psutil",  # core modules
        "cleanup_globals", "cleanup_session_state" # utility functions
    }

    for var in dir(mod):
        if var.startswith("__") or var in protected_vars:
            continue
        val = getattr(mod, var)
        if isinstance(val, (pd.DataFrame, np.ndarray, dict, list)):
            delattr(mod, var)
    gc.collect()



###############################################################################
# 5. File paths & feature definitions
script_loc  = os.path.dirname(os.path.abspath(__file__))
model_path  = os.path.join(script_loc, "tuned_voting_model.pkl")
prep_path   = os.path.join(script_loc, "preprocessor.pkl")
thresh_path = os.path.join(script_loc, "tuned_best_thresholds.json")

feature_names         = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
onehot_feats          = ["Type_H", "Type_L", "Type_M"]
updated_feature_names = feature_names + onehot_feats
target_names          = ["TWF", "HDF", "PWF", "OSF"]

label_map = {0: "No Failure", 1: "Failure"}
failure_descriptions = {
    "TWF": "Tool Wear Failure: Gradual tool degradation due to friction or usage.",
    "HDF": "Heat Dissipation Failure: Excessive heat not properly dissipated causing malfunction.",
    "PWF": "Power Failure: Issues related to power supply or fluctuations.",
    "OSF": "Overstrain Failure: Stress beyond mechanical tolerance causing breakdown."
}
demo_input  = {
    "Type": "M",
    "Air temperature [K]": 300.8,
    "Process temperature [K]": 309.4,
    "Rotational speed [rpm]": 1342,
    "Torque [Nm]": 62.4,
    "Tool wear [min]": 113
}
demo_target = "HDF"

###############################################################################
# 6. Cache model, preprocessor, thresholds, and test data
@st.cache_resource
def load_model():
    return load(model_path)

@st.cache_resource
def load_preprocessor():
    return load(prep_path)

@st.cache_resource
def load_thresholds():
    with open(thresh_path, "r") as f:
        return json.load(f)

@st.cache_data
def load_test_data():
    X = pd.read_csv(os.path.join(script_loc, "X_test.csv"))
    y = pd.read_csv(os.path.join(script_loc, "y_test.csv"))
    X = X[feature_names + ["Type"]]
    y = y[target_names]
    return X, y

@st.cache_data
def get_background():
    X_test_transformed = preprocessor.transform(X_test)
    return shap.kmeans(X_test_transformed, k=20)

@st.cache_resource
def get_explainer(_model, _background, target_idx):
    def predict_fn(arr):
        p = _model.predict_proba(arr)
        return p[target_idx][:, 1] if isinstance(p, list) else p[:, 1]
    return shap.KernelExplainer(predict_fn, _background)

model, preprocessor, thresholds = load_model(), load_preprocessor(), load_thresholds()
X_test, y_test = load_test_data()

###############################################################################
# 7. Prediction helper
def predict_one(input_dict, target_idx):
    df_in  = pd.DataFrame([input_dict])
    X_proc = preprocessor.transform(df_in)
    probas = model.predict_proba(X_proc)
    p      = (probas[target_idx][:, 1][0] if isinstance(probas, list)
              else probas[:, 1][0])
    return {
        "Prediction": int(p >= thresholds[target_names[target_idx]]),
        "Probability": p,
        "Threshold": thresholds[target_names[target_idx]]
    }

###############################################################################
# 8. Sidebar navigation & session-state reset
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predictions", "Model Metrics", "Sensor Influence", "About"])

if "prev_page" not in st.session_state:
    st.session_state.prev_page = page
elif st.session_state.prev_page != page:
    cleanup_session_state()
    cleanup_globals()
    gc.collect()
    st.session_state.prev_page = page
    for key in [
        "has_submitted", "last_input", "last_target",
        "demo_mode", "prediction_result", "shap_image"
    ]:
        st.session_state.pop(key, None)
    st.session_state.prev_page = page

###############################################################################
# --- Home Page ---
if page == "Home":
    st.title("AIML Capstone Project")
    st.set_page_config(page_title="Home", layout="wide")
    st.subheader("Using AI/ML for Predictive Maintenance in Industry")

    img = Image.open(os.path.join(script_loc, "NCI_Logo_white.png"))
    st.image(img, width=600)

###############################################################################
# --- Predictions Page ---
elif page == "Predictions":
    st.set_page_config(page_title="Predictions", layout="wide")
    st.title("Make Predictions")
    st.header("Enter Sensor Data")

    # Session-state defaults
    defaults = {
        "has_submitted": False,
        "last_input": {},
        "last_target": 0,
        "demo_mode": False,
        "prediction_result": None
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    if "force_demo_off" in st.session_state:
        st.session_state.demo_mode = False
        del st.session_state.force_demo_off

    st.sidebar.checkbox("Simulate User Input", key="demo_mode")
    container = st.container()

    def run_prediction():
        st.session_state.prediction_result = predict_one(
            st.session_state.last_input,
            st.session_state.last_target
        )
        st.session_state.has_submitted = True

    def reset_form():
        st.session_state.has_submitted = False
        st.session_state.prediction_result = None
        gc.collect()

    if not st.session_state.has_submitted:
        with container, st.form("prediction_form"):
            user_input = {}
            for feat in feature_names:
                mean_val = float(X_test[feat].mean())
                default  = (demo_input[feat]
                            if st.session_state.demo_mode else mean_val)
                user_input[feat] = st.number_input(feat, value=default)

            types       = ["H", "L", "M"]
            default_ty  = (demo_input["Type"]
                           if st.session_state.demo_mode else "M")
            user_input["Type"] = st.selectbox(
                "Machine Type", types, index=types.index(default_ty)
            )

            default_idx = (target_names.index(demo_target)
                           if st.session_state.demo_mode else 0)
            selected_target = st.selectbox(
                "Select failure type to predict & explain:",
                options=list(range(len(target_names))),
                format_func=lambda i: target_names[i],
                index=default_idx
            )

            with st.expander("ℹ️ What do the failure types mean?"):
                for tla, desc in failure_descriptions.items():
                    st.write(f"**{tla}**: {desc}")

            if st.form_submit_button("Predict"):
                st.session_state.last_input  = user_input
                st.session_state.last_target = selected_target
                run_prediction()
                st.rerun()
                gc.collect()

    else:
        with container:
            res = st.session_state.prediction_result
            tgt = st.session_state.last_target

            st.markdown(f"### {target_names[tgt]} Prediction")
            st.write(f"- Prediction: **{label_map[res['Prediction']]}**")
            st.write(f"- Probability: **{res['Probability']:.3f}**")
            st.write(f"- Threshold: **{res['Threshold']:.3f}**")

            # Compute SHAP values single-threaded
            df_in      = pd.DataFrame([st.session_state.last_input])
            X_proc     = preprocessor.transform(df_in)
            background = get_background()

            def predict_pos_prob(arr: np.ndarray) -> np.ndarray:
                p = model.predict_proba(arr)
                return (p[tgt][:, 1] if isinstance(p, list) else p[:, 1])

            with threadpool_limits(limits=1, user_api="blas"):
                explainer = get_explainer(model, background, tgt)

                shap_vals  = explainer.shap_values(X_proc, nsamples=20)

            raw = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
            exp = shap.Explanation(
                values=raw[0],
                base_values=explainer.expected_value,
                data=X_proc[0],
                feature_names=updated_feature_names
            )

            st.markdown("### Key Influential Features")
            top_idxs = np.argsort(np.abs(exp.values))[::-1][:3]
            for i in top_idxs:
                arrow = "⬆️" if exp.values[i] > 0 else "⬇️"
                st.write(
                    f"- **{exp.feature_names[i]}**: "
                    f"{arrow} (SHAP: `{exp.values[i]:.3f}`)"
                )

            shap_df = pd.DataFrame({
                'feature':    exp.feature_names,
                'shap_value': exp.values
            })
            shap_df['abs_shap'] = shap_df['shap_value'].abs()
            shap_df          = shap_df.sort_values('abs_shap', ascending=True)

            fig_shap = px.bar(
                shap_df,
                x='shap_value',
                y='feature',
                orientation='h',
                title=(
                    f"{target_names[tgt]} ➤ {label_map[res['Prediction']]} "
                    f"(Prob {res['Probability']:.2f}, Th {res['Threshold']:.3f})"
                ),
                labels={'shap_value': 'SHAP Value', 'feature': 'Feature'},
                color='shap_value',
                color_continuous_scale='Viridis'
            )
            c1, c2, c3 = st.columns([1, 8, 1])
            with c2:
                st.plotly_chart(fig_shap, use_container_width=True)
            del shap_vals, explainer, background, X_proc, df_in, exp, shap_df, fig_shap
            gc.collect()
            st.button("🔁 Try Again", on_click=reset_form)


###############################################################################
# --- Model Metrics Page ---
elif page == "Model Metrics":
    st.set_page_config(page_title="Model Metrics", layout="wide")
    st.title("Model Performance Dashboard")
    st.header("Primary Focus: Recall Scores")

    X_array = preprocessor.transform(X_test)
    def evaluate(model, X, y, targets=None, thresholds=None):
        if targets is None:
            targets = y.columns
        performance = {}
        proba_list  = model.predict_proba(X)
        for i, t in enumerate(targets):
            y_true    = y[t]
            y_prob    = proba_list[i][:, 1]
            thresh    = thresholds.get(t, 0.5)
            y_pred    = (y_prob >= thresh).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall    = recall_score(y_true, y_pred, zero_division=0)
            f1        = f1_score(y_true, y_pred, zero_division=0)
            performance[t] = [precision, recall, f1]
        return pd.DataFrame(performance, index=["Precision", "Recall", "F1-score"]).T

    df_metrics = evaluate(model, X_array, y_test, target_names, thresholds)

    with st.expander("ℹ️ Why Prioritise Recall?"):
        st.markdown("""
            In predictive maintenance, recall is critical for catching as many actual failure
            cases as possible. A missed failure could result in unexpected downtime or safety hazards.
        """)

    st.subheader("Recall by Failure Mode")
    # Create one column per target
    cols = st.columns(len(target_names))

    # Loop through each target and place its metric in a column
    for col, t in zip(cols, target_names):
        with col:
            st.markdown(f"**{t}**")
            st.metric("Recall", f"{df_metrics.loc[t, 'Recall']:.3f}")

    st.subheader("Recall Distribution")
    fig_recall = px.bar(
        df_metrics.reset_index(),
        x='index',
        y='Recall',
        title="Recall by Failure Mode",
        labels={'index': 'Failure Mode', 'Recall': 'Recall Score'},
        range_y=[0, 1],
        color='Recall',
        color_continuous_scale='Viridis'
    )
    c1, c2, c3 = st.columns([2, 6, 2])
    with c2:
        st.plotly_chart(fig_recall, use_container_width=True)


    st.subheader("Supporting Metrics (Precision & F1-score)")
    for t in target_names:
        st.markdown(f"**{t}**")
        c1, c2 = st.columns(2)
        c1.metric("Precision", f"{df_metrics.loc[t, 'Precision']:.3f}")
        c2.metric("F1-score", f"{df_metrics.loc[t, 'F1-score']:.3f}")

    st.subheader("Averages Across All Failure Modes")
    avg = df_metrics.mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Recall", f"{avg['Recall']:.3f}")
    c2.metric("Avg Precision", f"{avg['Precision']:.3f}")
    c3.metric("Avg F1-score", f"{avg['F1-score']:.3f}")
    del df_metrics, avg, X_array
    gc.collect()

###############################################################################
# --- Sensor Influence Page ---
elif page == "Sensor Influence":
    st.set_page_config(page_title="Sensor Influence", layout="centered")
    st.title("Sensor Influence Explorer")
    st.markdown("""
        Visualise how the different sensor types can influence each type of failure.
    """)
    selected_target = st.selectbox("Choose failure type:", target_names)
    with st.expander("ℹ️ What do the failure types mean?"):
        for tla, desc in failure_descriptions.items():
            st.write(f"**{tla}**: {desc}")
    tgt_idx         = target_names.index(selected_target)
    single_clf      = model.estimators_[tgt_idx]
    X_proc          = preprocessor.transform(X_test)

    # Force permutation_importance onto 1 job + 1 BLAS/OpenMP thread
    with threadpool_limits(limits=1, user_api="blas"):
        pi = permutation_importance(
            single_clf,
            X_proc,
            y_test[selected_target].values,
            n_repeats=5,
            random_state=42,
            n_jobs=1,              # limit to one worker
            scoring='recall_macro'
        )

    imp_df = (
        pd.DataFrame({
            'feature':    updated_feature_names,
            'importance': pi.importances_mean
        })
        .sort_values(by='importance', ascending=True)
    )

    fig_imp = px.bar(
        imp_df,
        x='importance',
        y='feature',
        orientation='h',
        title=f"Permutation Importance – {selected_target}",
        labels={'importance': 'Mean Influence on Failures', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    del pi, imp_df, fig_imp, X_proc, single_clf
    gc.collect()

###############################################################################
# --- About Page ---
else:
    st.title("About")
    st.set_page_config(page_title="About", layout="wide")
    st.markdown("""
        Predictive Maintenance App using a Voting Classifier with SHAP explanations.
        Models: Logistic Regression, Random Forest, AdaBoost, CatBoost.
        Written in Python 3.11.9 
    """)
    st.markdown("Built by **Eamonn Browne** | [Email](mailto:x23309709@student.ncirl.ie)")
    st.markdown("Supervisor: **Jaswinder Singh** | [Email](mailto:Jaswinder.Singh@ncirl.ie)")
    st.markdown("""
    - Version: 1.5.2 
    - Updated: August 2025
    - License: MIT licence 
    """)