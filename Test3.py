#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Purchase Prediction + XAI", layout="wide")

st.title("🛒 Prediksi Pembelian Visitor (Revenue) + XAI")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("shop_intent.csv")


# In[ ]:


# ===============================
# ENCODING CATEGORICAL
# ===============================
cat_cols = ["Month", "VisitorType", "Weekend", "Revenue"]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

y = df["Revenue"]
X = df.drop(columns=["Revenue"])


# In[ ]:


# ===============================
# TRAIN MODEL
# ===============================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)
model.fit(X, y)

st.success("✅ Model trained successfully!")


# In[ ]:


# ===============================
# GLOBAL XAI
# ===============================
st.header("🧠 Global XAI — Faktor yang Mempengaruhi Pembelian")

importances = model.feature_importances_

fi_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots()
ax.barh(fi_df["Feature"], fi_df["Importance"])
ax.invert_yaxis()
ax.set_title("Global Feature Importance")
st.pyplot(fig)

st.dataframe(fi_df)


# In[ ]:


# ===============================
# FEATURE SELECTION
# ===============================
st.subheader("✂️ Feature Selection Otomatis")

threshold = st.slider("Minimum importance threshold", 0.0, 0.2, 0.02)

selected_features = fi_df[fi_df["Importance"] >= threshold]["Feature"].tolist()

st.write("✅ Fitur yang dipakai model:")
st.write(selected_features)

X_selected = X[selected_features]

model_fs = RandomForestClassifier(n_estimators=300, random_state=42)
model_fs.fit(X_selected, y)


# In[ ]:


# ===============================
# INPUT UI
# ===============================
st.header("✍️ Input Data Visitor")

def ui_number(name, default=0.0):
    return st.number_input(name, value=float(default))

input_data = {}

for col in X.columns:
    if col == "Month":
        input_data[col] = st.selectbox("Month", encoders["Month"].classes_)
    elif col == "VisitorType":
        input_data[col] = st.selectbox("VisitorType", encoders["VisitorType"].classes_)
    elif col == "Weekend":
        input_data[col] = st.selectbox("Weekend", encoders["Weekend"].classes_)
    else:
        input_data[col] = st.number_input(col, value=float(df[col].median()))

# Encode categorical input
for col in ["Month", "VisitorType", "Weekend"]:
    input_data[col] = encoders[col].transform([input_data[col]])[0]


# In[ ]:


# ===============================
    # LOCAL XAI
    # ===============================
    st.header("🔍 Local Explanation — Kenapa hasilnya seperti ini?")

    perm = permutation_importance(
        model_fs,
        X_selected,
        y,
        n_repeats=10,
        random_state=42
    )

    local_imp = pd.DataFrame({
        "Feature": selected_features,
        "Importance": perm.importances_mean
    }).sort_values(by="Importance", ascending=False)

    fig2, ax2 = plt.subplots()
    ax2.barh(local_imp["Feature"], local_imp["Importance"])
    ax2.invert_yaxis()
    ax2.set_title("Local Feature Importance")
    st.pyplot(fig2)

    st.dataframe(local_imp)

    st.subheader("🏆 Faktor Paling Menentukan:")

    for i, row in local_imp.head(5).iterrows():
        st.write(f"• **{row['Feature']}** (impact: {row['Importance']:.4f})")

