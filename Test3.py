#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Purchase Prediction + XAI", layout="wide")
st.title("🛒 Prediksi Pembelian Visitor (Revenue) + XAI")

# ===============================
# SESSION DATASET
# ===============================
if "df" not in st.session_state:
    st.session_state.df = None


# In[ ]:


# ===============================
# 1️⃣ UPLOAD CSV
# ===============================
st.header("1️⃣ Upload Dataset CSV")

uploaded = st.file_uploader("Upload CSV (Online Shoppers)", type=["csv"])

if uploaded:
    new_df = pd.read_csv(uploaded)

    if st.session_state.df is None:
        st.session_state.df = new_df
    else:
        st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)

    st.success(f"✅ Dataset sekarang berisi {len(st.session_state.df)} baris")

# ===============================
# STOP IF NO DATA
# ===============================
if st.session_state.df is None:
    st.warning("⚠️ Silakan upload dataset dulu")
    st.stop()

df = st.session_state.df.copy()

# ===============================
# ENCODING
# ===============================
cat_cols = ["Month", "VisitorType", "Weekend", "Revenue"]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le


# In[ ]:


# ===============================
# 2️⃣ INPUT MANUAL TRAINING
# ===============================
st.header("2️⃣ Tambah Data Manual ke Dataset (Training)")

with st.expander("➕ Tambah 1 Baris Data Training"):

    manual_row = {}

    for col in df.columns:
        if col in ["Month", "VisitorType", "Weekend", "Revenue"]:
            manual_row[col] = st.selectbox(f"{col}", encoders[col].classes_)
        else:
            manual_row[col] = st.number_input(col, value=float(df[col].median()))

    if st.button("➕ Tambahkan ke Dataset & Retrain"):
        for col in ["Month", "VisitorType", "Weekend", "Revenue"]:
            manual_row[col] = encoders[col].transform([manual_row[col]])[0]

        st.session_state.df = pd.concat(
            [st.session_state.df, pd.DataFrame([manual_row])],
            ignore_index=True
        )

        st.success("✅ Data ditambahkan, model akan retrain")
        st.rerun()


# In[ ]:


# ===============================
# TARGET & FEATURE
# ===============================
y = df["Revenue"]
X = df.drop(columns=["Revenue"])

if y.nunique() < 2:
    st.error("❌ Dataset hanya punya 1 kelas. Tambahkan data kelas lain.")
    st.stop()

# ===============================
# TRAIN MODEL
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ===============================
# 3️⃣ HASIL MODEL
# ===============================
st.header("3️⃣ Performa Model")

st.success(f"🎯 Akurasi Model: {acc*100:.2f}%")


# In[ ]:


# ===============================
# GLOBAL FEATURE IMPORTANCE
# ===============================
st.subheader("🧠 Global Feature Importance")

fi_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots()
ax.barh(fi_df["Feature"], fi_df["Importance"])
ax.invert_yaxis()
st.pyplot(fig)

st.dataframe(fi_df)


# In[ ]:


# ===============================
# 4️⃣ INPUT MANUAL TEST
# ===============================
st.header("4️⃣ Input Data untuk Test Prediksi")

input_test = {}

for col in X.columns:
    if col in ["Month", "VisitorType", "Weekend"]:
        input_test[col] = st.selectbox(f"{col} (Test)", encoders[col].classes_)
    else:
        input_test[col] = st.number_input(f"{col} (Test)", value=float(df[col].median()))

# encode
for col in ["Month", "VisitorType", "Weekend"]:
    input_test[col] = encoders[col].transform([input_test[col]])[0]

input_pred = pd.DataFrame([input_test])


# In[ ]:


# ===============================
# 5️⃣ PREDIKSI
# ===============================
if st.button("🔮 Predict Purchase"):

    pred = model.predict(input_pred)[0]
    prob = model.predict_proba(input_pred)[0][pred]

    if pred == 1:
        st.success(f"🛒 PREDIKSI: AKAN BELI ({prob*100:.2f}%)")
    else:
        st.warning(f"❌ PREDIKSI: TIDAK BELI ({prob*100:.2f}%)")

    # ===============================
    # 6️⃣ XAI SEDERHANA
    # ===============================
    st.header("6️⃣ XAI: Faktor Dominan")

    top_feats = fi_df.head(5)

    for _, row in top_feats.iterrows():
        st.write(f"• **{row['Feature']}** (importance: {row['Importance']:.4f})")

    # ===============================
    # 7️⃣ SHAP LOCAL
    # ===============================
    st.header("7️⃣ SHAP: Kenapa Prediksi Ini Terjadi?")

    shap_exp_local = explainer(input_pred)

    pred_class = int(pred)

    shap_vals = shap_exp_local.values[0][:, pred_class]
    base_val = shap_exp_local.base_values[0][pred_class]

    shap_explanation = shap.Explanation(
        values=shap_vals,
        base_values=base_val,
        data=input_pred.iloc[0],
        feature_names=input_pred.columns
    )

    fig_local, ax_local = plt.subplots()
    shap.plots.waterfall(shap_explanation, show=False)
    st.pyplot(fig_local)

