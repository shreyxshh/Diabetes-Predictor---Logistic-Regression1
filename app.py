import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_utils import FEATURES, GENDER_CODES, SMOKING_CODES, load_artifacts

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.verdict {padding: 1.1rem 1.3rem; border-radius: 12px; font-size: 1.25rem; font-weight: 600;}
.verdict.high {background: #fdecea; color: #b42318; border: 1px solid #f5c2bd;}
.verdict.low  {background: #e8f6ee; color: #146c43; border: 1px solid #b7e1c9;}
.small {color: #6b7280; font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model…")
def get_artifacts():
    return load_artifacts()


try:
    clf, scaler, metrics = get_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

LABELS = {"gender": "Gender", "age": "Age", "hypertension": "Hypertension",
          "heart_disease": "Heart disease", "smoking_history": "Smoking history",
          "bmi": "BMI", "HbA1c_level": "HbA1c level",
          "blood_glucose_level": "Blood glucose"}


def gauge(prob: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number={"suffix": "%", "font": {"size": 44}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#1f2937", "thickness": 0.25},
               "steps": [{"range": [0, 30], "color": "#bbf7d0"},
                         {"range": [30, 60], "color": "#fde68a"},
                         {"range": [60, 100], "color": "#fecaca"}],
               "threshold": {"line": {"color": "black", "width": 3},
                             "thickness": 0.8, "value": 50}}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=0))
    return fig


def contribution_chart(contrib: pd.Series) -> go.Figure:
    contrib = contrib.sort_values()
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in contrib]
    fig = go.Figure(go.Bar(x=contrib.values, y=[LABELS[i] for i in contrib.index],
                           orientation="h", marker_color=colors))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Push toward diabetic  →", plot_bgcolor="white")
    return fig


st.title("🩺 Diabetes Risk Predictor")
st.caption("Logistic Regression model trained on 100,000 patient records · "
           "built with scikit-learn & Streamlit")

tab_predict, tab_about = st.tabs(["🔮 Predict", "📘 About the model"])

# ---------------------------------------------------------------- Predict tab
with tab_predict:
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("Patient details")
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Gender", list(GENDER_CODES))
        age = c2.number_input("Age (years)", 1, 100, 45)
        c3, c4 = st.columns(2)
        hypertension = c3.radio("Hypertension", ["No", "Yes"], horizontal=True)
        heart = c4.radio("Heart disease", ["No", "Yes"], horizontal=True)
        smoking = st.selectbox("Smoking history", list(SMOKING_CODES), index=4)
        bmi = st.slider("BMI", 10.0, 70.0, 27.0, 0.1)
        hba1c = st.slider("HbA1c level (%)", 3.5, 9.0, 5.5, 0.1)
        glucose = st.slider("Blood glucose level (mg/dL)", 80, 300, 120, 5)
        go_btn = st.button("Predict", type="primary", width="stretch")

    row = pd.DataFrame([{
        "gender": GENDER_CODES[gender], "age": age,
        "hypertension": int(hypertension == "Yes"),
        "heart_disease": int(heart == "Yes"),
        "smoking_history": SMOKING_CODES[smoking], "bmi": bmi,
        "HbA1c_level": hba1c, "blood_glucose_level": glucose}])[FEATURES]

    with right:
        st.subheader("Result")
        if not go_btn and "predicted" not in st.session_state:
            st.info("Fill in the details and press **Predict**.")
        else:
            st.session_state["predicted"] = True
            scaled = scaler.transform(row)
            prob = float(clf.predict_proba(scaled)[0][1])
            is_diabetic = prob >= 0.5

            if is_diabetic:
                st.markdown('<div class="verdict high">⚠️ Likely diabetic</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict low">✅ Likely not diabetic</div>',
                            unsafe_allow_html=True)
            st.plotly_chart(gauge(prob), width="stretch")
            m1, m2 = st.columns(2)
            m1.metric("Diabetic probability", f"{prob*100:.1f}%")
            m2.metric("Non-diabetic probability", f"{(1-prob)*100:.1f}%")

            st.markdown("**What drove this prediction?**")
            contrib = pd.Series(clf.coef_[0] * scaled[0], index=FEATURES)
            st.plotly_chart(contribution_chart(contrib), width="stretch")
            st.markdown('<p class="small">Each bar is the feature’s coefficient × its '
                        'scaled value — red pushes toward diabetic, green away.</p>',
                        unsafe_allow_html=True)

    st.divider()
    st.caption("⚕️ Educational project only — not a medical diagnosis. "
               "Please consult a healthcare professional.")

# ----------------------------------------------------------------- About tab
with tab_about:
    st.subheader("How it works")
    st.markdown("""
1. **Encode** text columns (gender, smoking history) into numbers
2. **Split** 80% train / 20% test (`random_state=42`)
3. **Scale** features with `StandardScaler`
4. **Train** `LogisticRegression(class_weight="balanced")`
5. **Predict** — probability ≥ 50% is labelled diabetic

`class_weight="balanced"` makes the model prioritise catching diabetic patients
(recall), which matters in medical screening — but it also inflates the
probabilities, so treat them as a risk score rather than a literal chance.
""")
    if metrics:
        st.subheader("Test-set performance")
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{metrics['accuracy']:.2f}")
        b.metric("Precision", f"{metrics['precision']:.2f}")
        c.metric("Recall", f"{metrics['recall']:.2f}")
        d.metric("F1 score", f"{metrics['f1']:.2f}")
        st.caption(f"Evaluated on {metrics['n_test']:,} held-out patients.")

    st.subheader("Model coefficients")
    coefs = pd.Series(clf.coef_[0], index=[LABELS[f] for f in FEATURES]).sort_values()
    fig = go.Figure(go.Bar(x=coefs.values, y=coefs.index, orientation="h",
                           marker_color=["#ef4444" if v > 0 else "#22c55e" for v in coefs]))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
    st.plotly_chart(fig, width="stretch")
