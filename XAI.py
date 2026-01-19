#!/usr/bin/env python
# coding: utf-8

# In[14]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import shap
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title="XAI PayLater Demo", layout="wide")

# ==============================
# 1. Load Dataset & Mappings
# ==============================
data = pd.read_csv('data.csv')

gender_mapping = {
    'Male': 0,
    'Female': 1
}

education_mapping = {
    'High School': 0,
    'Bachelor': 1,
    'Master': 2,
    'PhD': 3
}

payment_mapping = {
    'Bad': 0,
    'Average': 1,
    'Good': 2
}

employment_mapping = {
    'Unemployed': 0,
    'Employed': 1,
    'Self-Employed': 2
}

residence_mapping = {
    'Mortgaged': 0,
    'Rented': 1,
    'Owned': 2
}

marriage_mapping = {
    'Single': 0,
    'Married': 1,
    'Divorced': 2
}

# Apply the mapping manually
data["Gender"] = [gender_mapping[item] for item in data["Gender"]]
data["Education"] = [education_mapping[item] for item in data["Education"]]
data["Payment_History"] = [payment_mapping[item] for item in data["Payment_History"]]
data["Employment_Status"] = [employment_mapping[item] for item in data["Employment_Status"]]
data["Residence_Type"] = [residence_mapping[item] for item in data["Residence_Type"]]
data["Marital_Status"] = [marriage_mapping[item] for item in data["Marital_Status"]]

X = data.drop("Creditworthiness", axis=1)
y = data["Creditworthiness"]


# In[15]:


# ==============================
# 2. Train Random Forest Model
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_train, y_train)


# In[16]:


# ==============================
# 3. UI Header
# ==============================
st.title("Explainable AI (XAI) – PayLater Decision System")
st.markdown("""
Peserta dapat **mengisi data sendiri** atau **melihat prediksi dari dataset**,  
lalu melihat **prediksi AI + penjelasannya**.
""")


# In[17]:


# ==============================
# 4. Show Prediction Results from Dataset
# ==============================
st.subheader("📊 Prediction Results from Dataset")

# Predict on the entire dataset
dataset_predictions = model.predict(X)
dataset_probabilities = model.predict_proba(X)[:, 1]

result_df = X.copy()
result_df["Actual"] = y
result_df["Predicted"] = dataset_predictions
result_df["Confidence (%)"] = np.round(dataset_probabilities * 100, 2)

result_df["Predicted"] = result_df["Predicted"].map(
    {1: "✅ Approved", 0: "❌ Rejected"}
)
result_df["Actual"] = result_df["Actual"].map(
    {1: "✅ Approved", 0: "❌ Rejected"}
)

st.dataframe(result_df.head(20))

accuracy = (dataset_predictions == y).mean()
st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")


# In[7]:


# ==============================
# 5. Manual Input Section
# ==============================
st.markdown("---")
st.subheader("🔹 Manual Data Input for New Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 15, 80, 40)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    education = st.selectbox("Education", options=["High School", "Bachelor", "Master", "PhD"])
    paymenthistory = st.selectbox("Payment History", options=["Good", "Average", "Bad"])
    
with col2:
    monthly_income = st.slider("Income", 50_000, 150_000, 100_000)
    monthly_expense = st.slider("Debt", 10_000, 50_000, 25_000)
    employmentstatus = st.selectbox("Employment Status", options=["Employed", "Self-Employed", "Unemployed"])
    residence = st.selectbox("Residence Type", options=["Owned", "Rented", "Mortgaged"])

with col3:
    credit_cards_count = st.slider("Num_Credit_Cards", 0, 10, 5)
    credit_score = st.slider("Credit_Score", 500, 800, 650)
    marital = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
    loan_amount = st.slider("Loan Amount", 500, 50_000, 5_000)
    loan_term = st.slider("Loan Term", 12, 60, 48)

# Encode manual input
gender_encoded = gender_mapping[gender]
education_encoded = education_mapping[education]
paymenthistory_encoded = payment_mapping[paymenthistory]
employmentstatus_encoded = employment_mapping[employmentstatus]
residence_encoded = residence_mapping[residence]
marital_encoded = marriage_mapping[marital]

# Create DataFrame for manual input (match your dataset column names)
input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender_encoded,
    "Education": education_encoded,
    "Income": monthly_income,
    "Debt": monthly_expense,
    "Credit_Score": credit_score,
    "Loan_Amount": loan_amount,
    "Loan_Term": loan_term
    "Num_Credit_Cards": credit_cards_count,
    "Payment_History": paymenthistory_encoded,
    "Employment_Status": employmentstatus_encoded,
    "Residence_Type": residence_encoded,
    "Marital_Status": marital_encoded,
}])


# In[8]:


# ==============================
# 6. Prediction on Manual Input
# ==============================
st.subheader("🔮 Your Prediction Result")

manual_prediction = model.predict(input_df)[0]
manual_probability = model.predict_proba(input_df)[0, 1]

manual_result_df = input_df.copy()
manual_result_df["Approval"] = manual_prediction
manual_result_df["Confidence (%)"] = np.round(manual_probability * 100, 2)

manual_result_df["Approval"] = manual_result_df["Approval"].map(
    {1: "✅ Approved", 0: "❌ Rejected"}
)

st.dataframe(manual_result_df)


# In[9]:


# ==============================
# 7. Global Feature Importance
# ==============================
st.markdown("---")
st.subheader("📊 Global Explanation – Feature Importance")

fi_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(fi_df.set_index("Feature"))


# In[18]:


# ==============================
# 8. SHAP (Local Explanation)
# ==============================
st.subheader("🧠 Local Explanation – SHAP (for your manual input)")

shap_explainer = shap.TreeExplainer(model)
shap_values = shap_explainer.shap_values(input_df)

fig, ax = plt.subplots()
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0][0],
        base_values=shap_explainer.expected_value[1],
        data=input_df.iloc[0],
        feature_names=X.columns
    ),
    show=False
)
st.pyplot(fig)


# In[19]:


# ==============================
# 9. LIME (Local Explanation)
# ==============================
st.subheader("🧪 Local Explanation – LIME (for your manual input)")

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns,
    class_names=["Rejected", "Approved"],
    mode="classification"
)

lime_exp = lime_explainer.explain_instance(
    input_df.iloc[0].values,
    model.predict_proba
)

fig_lime = lime_exp.as_pyplot_figure()
st.pyplot(fig_lime)


# In[20]:


# ==============================
# 10. Reflection Prompt
# ==============================
st.info("""
**Reflection Questions:**
- Which feature influenced the decision the most?
- Do you agree with the AI decision?
- What would you change to get a different outcome?
""")


# In[ ]:





# In[ ]:





# In[ ]:




