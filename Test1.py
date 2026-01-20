#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Engagement Predictor + XAI + SHAP", layout="wide")

st.title("📊 Prediksi Viralitas Konten (Multi Upload + Manual Input + XAI + SHAP)")

# ===============================
# SESSION STATE
# ===============================
if "df" not in st.session_state:
    st.session_state.df = None

# ===============================
# UPLOAD CSV
# ===============================
st.header("📤 Upload Dataset CSV (Bisa berkali-kali, data akan digabung)")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    new_df = pd.read_csv(uploaded_file)

    if st.session_state.df is None:
        st.session_state.df = new_df.copy()
    else:
        st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)

    st.success(f"✅ Dataset ditambahkan! Total baris sekarang: {len(st.session_state.df)}")

# ===============================
# SHOW DATASET
# ===============================
if st.session_state.df is not None:
    st.subheader("📄 Dataset Saat Ini")
    st.write(f"Total rows: {len(st.session_state.df)}")
    st.dataframe(st.session_state.df.head(50))


# In[ ]:


# ===============================
# ADD MANUAL ROW
# ===============================
if st.session_state.df is not None:
    st.header("➕ Tambah 1 Data Manual ke Dataset")

    df = st.session_state.df

    input_row = {}

    for col in df.columns:
        if col == "Engagement_Label":
            continue
        if df[col].dtype == object:
            input_row[col] = st.text_input(f"{col}", value=str(df[col].mode()[0]))
        else:
            input_row[col] = st.number_input(f"{col}", value=float(df[col].median()))

    if st.button("➕ Tambahkan ke Dataset"):
        new_row = pd.DataFrame([input_row])
        st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
        st.success("✅ Data baru ditambahkan ke dataset!")


# In[ ]:


# ===============================
# TRAINING PIPELINE
# ===============================
if st.session_state.df is not None and len(st.session_state.df) >= 5:

    st.header("🧠 Training Model Otomatis")

    df = st.session_state.df.copy()

    # ===============================
    # BASIC CLEANING
    # ===============================
    encoders = {}
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # ===============================
    # CREATE LABEL IF NEEDED
    # ===============================
    if "Engagement_Label" not in df.columns:
        if "Total Interactions" in df.columns:
            median_interaction = df["Total Interactions"].median()
            df["Engagement_Label"] = (df["Total Interactions"] > median_interaction).astype(int)
        else:
            st.error("❌ Dataset harus punya kolom 'Total Interactions' atau 'Engagement_Label'")
            st.stop()

    # ===============================
    # FEATURES
    # ===============================
    drop_cols = ["Engagement_Label"]
    if "Total Interactions" in df.columns:
        drop_cols.append("Total Interactions")

    X = df.drop(columns=drop_cols)
    y = df["Engagement_Label"]


# In[ ]:


# ===============================
    # TRAIN TEST SPLIT
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # ===============================
    # TRAIN MODEL
    # ===============================
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # ===============================
    # EVALUATION
    # ===============================
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"🎯 Akurasi Model: {acc*100:.2f}%")


# In[ ]:


# ===============================
 # GLOBAL FEATURE IMPORTANCE
 # ===============================
 st.header("🧠 Global XAI — Feature Importance")

 fi = pd.DataFrame({
     "Feature": X.columns,
     "Importance": model.feature_importances_
 }).sort_values(by="Importance", ascending=False)

 fig, ax = plt.subplots()
 ax.barh(fi["Feature"], fi["Importance"])
 ax.invert_yaxis()
 ax.set_title("Global Feature Importance")
 st.pyplot(fig)

 st.dataframe(fi)

 # ===============================
 # SHAP GLOBAL
 # ===============================
 st.header("🧠 SHAP Global Explanation")

 explainer = shap.TreeExplainer(model)
 shap_values = explainer.shap_values(X_train)

 fig_shap, ax_shap = plt.subplots()
 shap.summary_plot(shap_values[1], X_train, show=False)
 st.pyplot(fig_shap)


# In[ ]:


# ===============================
 # PREDICTION INPUT
 # ===============================
 st.header("🔮 Prediksi 1 Post Baru")

 input_pred = {}

 for col in X.columns:
     if col in encoders:
         val = st.selectbox(f"{col} (prediksi)", encoders[col].classes_)
         input_pred[col] = encoders[col].transform([val])[0]
     else:
         input_pred[col] = st.number_input(f"{col} (prediksi)", value=float(X[col].median()))

 if st.button("🔮 Predict Engagement"):

     input_df = pd.DataFrame([input_pred])

     pred = model.predict(input_df)[0]
     prob = model.predict_proba(input_df)[0][pred]

     if pred == 1:
         st.success(f"🔥 Prediksi: HIGH Engagement ({prob*100:.2f}%)")
     else:
         st.warning(f"❄️ Prediksi: LOW Engagement ({prob*100:.2f}%)")

     # ===============================
     # SHAP LOCAL
     # ===============================
     st.header("🔍 SHAP Local Explanation")

     shap_val_single = explainer.shap_values(input_df)

     fig_local, ax_local = plt.subplots()
     shap.waterfall_plot(
         shap.Explanation(
             values=shap_val_single[1][0],
             base_values=explainer.expected_value[1],
             data=input_df.iloc[0],
             feature_names=input_df.columns
         ),
         show=False
     )
     st.pyplot(fig_local)

