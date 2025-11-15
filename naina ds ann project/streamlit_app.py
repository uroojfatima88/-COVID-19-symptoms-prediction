import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from tensorflow.keras.models import load_model

st.set_page_config(page_title="COVID Severity Predictor", layout="centered")

st.markdown(
    """
    <h1 style='text-align:center;color:#2c2c54;'>🌸 COVID Severity Predictor</h1>
    <p style='text-align:center;color:#6b6b83;'>Upload your trained model and make predictions instantly</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# Upload model + scaler
# ---------------------------------------------------
st.sidebar.header("📦 Upload Model & Scaler")

model_file = st.sidebar.file_uploader("Upload your model (.h5)", type=["h5"])
scaler_file = st.sidebar.file_uploader("Upload your scaler (.save, .pkl, .joblib)",
                                       type=["save", "pkl", "joblib"])

# ---------------------------------------------------
# Load model + scaler into session
# ---------------------------------------------------
if model_file and scaler_file:
    try:
        # Load model
        temp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
        temp_model.write(model_file.read())
        temp_model.flush()
        st.session_state["model"] = load_model(temp_model.name)

        # Load scaler
        temp_scaler = tempfile.NamedTemporaryFile(delete=False)
        temp_scaler.write(scaler_file.read())
        temp_scaler.flush()
        st.session_state["scaler"] = joblib.load(temp_scaler.name)

        st.success("Model & scaler loaded ✔")
    except Exception as e:
        st.error("Failed to load model/scaler: " + str(e))

# ---------------------------------------------------
# Upload dataset for prediction
# ---------------------------------------------------
st.header("📄 Upload Dataset for Prediction")
uploaded_csv = st.file_uploader("Upload CSV file to predict severity", type=["csv"])

# ---------------------------------------------------
# If model + scaler are loaded and CSV is uploaded...
# ---------------------------------------------------
if uploaded_csv and "model" in st.session_state and "scaler" in st.session_state:
    df = pd.read_csv(uploaded_csv)
    st.write("### Preview")
    st.dataframe(df.head())

    st.markdown("### 🔧 Processing input features…")

    # Identify categorical columns + encode
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    df_processed = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Make sure columns match model training
    expected = st.session_state["scaler"].feature_names_in_
    for col in expected:
        if col not in df_processed.columns:
            df_processed[col] = 0

    df_processed = df_processed[expected]

    # Scale using uploaded scaler
    scaled = st.session_state["scaler"].transform(df_processed)

    # Predict using uploaded model
    preds = st.session_state["model"].predict(scaled)
    final_preds = np.argmax(preds, axis=1)

    df["predicted_severity"] = final_preds

    st.success("Prediction complete! 🎉")
    st.dataframe(df.head())

    # Download results
    st.download_button(
        "Download Predictions CSV",
        df.to_csv(index=False).encode("utf-8"),
        "predictions.csv"
    )

# ---------------------------------------------------
# Single prediction
# ---------------------------------------------------
st.header("🔮 Single Prediction")

if "model" in st.session_state and "scaler" in st.session_state:

    st.write("Enter values manually:")

    user_input = {}
    for feature in st.session_state["scaler"].feature_names_in_:
        user_input[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict Single Case"):
        row = pd.DataFrame([user_input])
        scaled_row = st.session_state["scaler"].transform(row)
        proba = st.session_state["model"].predict(scaled_row)
        pred_class = int(np.argmax(proba))

        st.success(f"Predicted Severity Class: **{pred_class}**")
        st.write("Probability:", [float(x) for x in proba[0]])

else:
    st.info("Upload model + scaler to enable prediction.")

