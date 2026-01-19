#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Engagement Predictor + XAI", layout="wide")

st.title("📊 Prediksi Viralitas Konten (High vs Low Engagement) + XAI")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("dataset_Facebook.csv", delimiter=";")

# Encode Type
le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])

# Create Label
median_interaction = df["Total Interactions"].median()
df["Engagement_Label"] = (df["Total Interactions"] > median_interaction).astype(int)


# In[ ]:


# ===============================
# FEATURE SET
# ===============================
features = [
    "Type", "Category", "Post Month", "Post Weekday", "Post Hour", "Paid",
    "Lifetime Post Total Reach", "Lifetime Engaged Users", "comment", "like", "share"
]

X = df[features]
y = df["Engagement_Label"]


# In[ ]:


# ===============================
# TRAIN MODEL
# ===============================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42
)
model.fit(X, y)

st.success("✅ Model trained successfully!")


# In[ ]:


# ===============================
# GLOBAL FEATURE IMPORTANCE (XAI)
# ===============================
st.header("🧠 Global XAI — Faktor Penentu Viralitas")

importances = model.feature_importances_
fi_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Plot
fig, ax = plt.subplots()
ax.barh(fi_df["Feature"], fi_df["Importance"])
ax.invert_yaxis()
ax.set_title("Global Feature Importance (Random Forest)")
st.pyplot(fig)

st.dataframe(fi_df)


# In[ ]:


# ===============================
# FEATURE SELECTION (AUTO)
# ===============================
st.subheader("✂️ Feature Selection Otomatis")

threshold = st.slider("Minimum importance threshold", 0.0, 0.2, 0.03)

selected_features = fi_df[fi_df["Importance"] >= threshold]["Feature"].tolist()

st.write("✅ Fitur yang dipakai model:")
st.write(selected_features)

X_selected = df[selected_features]

# Retrain model with selected features
model_fs = RandomForestClassifier(n_estimators=300, random_state=42)
model_fs.fit(X_selected, y)


# In[ ]:


# ===============================
# INPUT UI
# ===============================
st.header("✍️ Input Data Post")

type_input = st.selectbox("Type", ["Photo", "Status"])
category = st.number_input("Category", 1, 3, 2)
post_month = st.number_input("Post Month", 1, 12, 12)
weekday = st.number_input("Post Weekday (1=Mon, 7=Sun)", 1, 7, 3)
hour = st.number_input("Post Hour", 0, 23, 10)
paid = st.selectbox("Paid Promotion?", [0, 1])

reach = st.number_input("Lifetime Post Total Reach", 0, 100000, 10000)
engaged = st.number_input("Lifetime Engaged Users", 0, 5000, 500)
comment = st.number_input("Comment", 0, 500, 10)
like = st.number_input("Like", 0, 5000, 100)
share = st.number_input("Share", 0, 500, 10)

# Encode type
type_encoded = le.transform([type_input])[0]

# Create full input dict
input_dict = {
    "Type": type_encoded,
    "Category": category,
    "Post Month": post_month,
    "Post Weekday": weekday,
    "Post Hour": hour,
    "Paid": paid,
    "Lifetime Post Total Reach": reach,
    "Lifetime Engaged Users": engaged,
    "comment": comment,
    "like": like,
    "share": share
}


# In[ ]:


# ===============================
# PREDICT + LOCAL XAI
# ===============================
if st.button("🔮 Predict Engagement"):

    input_df = pd.DataFrame([input_dict])

    # Only selected features
    input_selected = input_df[selected_features]

    prediction = model_fs.predict(input_selected)[0]
    prob = model_fs.predict_proba(input_selected)[0][prediction]

    if prediction == 1:
        st.success(f"🔥 Prediksi: HIGH Engagement ({prob*100:.2f}%)")
    else:
        st.warning(f"❄️ Prediksi: LOW Engagement ({prob*100:.2f}%)")


# In[ ]:


# ===============================
    # LOCAL EXPLANATION USING PERMUTATION
    # ===============================
    st.header("🔍 Local Explanation — Faktor Paling Berpengaruh untuk Post Ini")

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
    ax2.set_title("Local Feature Importance (Permutation Importance)")
    st.pyplot(fig2)

    st.dataframe(local_imp)

    # Highlight top factors
    st.subheader("🏆 Faktor Paling Menentukan:")

    for i, row in local_imp.head(3).iterrows():
        st.write(f"• **{row['Feature']}** (impact: {row['Importance']:.4f})")

