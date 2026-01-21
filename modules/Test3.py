#!/usr/bin/env python
# coding: utf-8

def run_purchase_app():
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import shap
    import numpy as np
    import os
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    st.title("🛒 Consumer Purchase Decision Prediction")
    st.header("\"Why Does the Customer Decide to Buy?\"")
    
    # ===============================
    # SESSION DATASET
    # ===============================
    if "df_purchase" not in st.session_state:
        st.session_state.df_purchase = None
    if "model_purchase" not in st.session_state:
        st.session_state.model_purchase = None
    if "encoders_purchase" not in st.session_state:
        st.session_state.encoders_purchase = None
    if "trained_purchase" not in st.session_state:
        st.session_state.trained_purchase = False
    if "initial_purchase_loaded" not in st.session_state:
        st.session_state.initial_purchase_loaded = False
    
    # ===============================
    # LOAD INITIAL CSV (ONLY ONCE)
    # ===============================
    INITIAL_CSV_PATH = "shop_intent.csv"  # Change this to your CSV file path
    
    if not st.session_state.initial_purchase_loaded:
        if os.path.exists(INITIAL_CSV_PATH):
            try:
                initial_df = pd.read_csv(INITIAL_CSV_PATH)
                st.session_state.df_purchase = initial_df
                st.session_state.initial_purchase_loaded = True
                st.info(f"✅ Initial dataset loaded: {len(initial_df)} rows from '{INITIAL_CSV_PATH}'")
            except Exception as e:
                st.error(f"❌ Error loading initial CSV: {e}")
                st.stop()
        else:
            st.error(f"❌ Initial CSV file not found: '{INITIAL_CSV_PATH}'")
            st.warning("⚠️ Please ensure the CSV file exists in the correct location.")
            st.stop()
    
    # ===============================
    # DISPLAY CURRENT DATASET INFO
    # ===============================
    if st.session_state.df_purchase is not None:
        st.info(f"📊 Current dataset size: **{len(st.session_state.df_purchase)} rows**")
        with st.expander("👁️ View Current Dataset (First 10 rows)"):
            st.dataframe(st.session_state.df_purchase.head(10))
    
    # ===============================
    # 1️⃣ UPLOAD ADDITIONAL CSV
    # ===============================
    st.header("1️⃣ Upload Additional Dataset CSV (Optional)")
    
    uploaded = st.file_uploader("Upload CSV to append to existing dataset", type=["csv"], key="tab3_purchase_uploader")
    
    if st.button("📥 Upload & Append to Dataset", key="tab3_upload_append_btn"):
        if uploaded is not None:
            new_df = pd.read_csv(uploaded)
            
            # Append to existing dataset
            st.session_state.df_purchase = pd.concat([st.session_state.df_purchase, new_df], ignore_index=True)
            
            st.success(f"✅ Added {len(new_df)} new rows! Total dataset: {len(st.session_state.df_purchase)} rows")
            st.rerun()
        else:
            st.warning("⚠️ Please select a file to upload")
    
    # ===============================
    # STOP IF NO DATA
    # ===============================
    if st.session_state.df_purchase is None:
        st.warning("⚠️ Upload the dataset to begin")
        st.stop()
    
    df = st.session_state.df_purchase.copy()
    
    # ===============================
    # ENCODING
    # ===============================
    cat_cols = ["Month", "VisitorType", "Weekend", "Revenue"]
    
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    # ===============================
    # 2️⃣ INPUT MANUAL TRAINING
    # ===============================
    st.header("2️⃣ Add new data into dataset (Training)")
    
    with st.expander("➕ Add 1 new row into dataset"):
    
        manual_row = {}
        
        col_idx = 0
        for col in df.columns:
            if col in ["Month", "VisitorType", "Weekend", "Revenue"]:
                manual_row[col] = st.selectbox(f"{col}", encoders[col].classes_, key=f"tab3_manual_{col_idx}")
            else:
                manual_row[col] = st.number_input(col, value=float(df[col].median()), key=f"tab3_manual_{col_idx}")
            col_idx += 1
    
        if st.button("➕ Tambahkan ke Dataset & Retrain", key="tab3_add_retrain_btn"):
            for col in ["Month", "VisitorType", "Weekend", "Revenue"]:
                manual_row[col] = encoders[col].transform([manual_row[col]])[0]
    
            st.session_state.df_purchase = pd.concat(
                [st.session_state.df_purchase, pd.DataFrame([manual_row])],
                ignore_index=True
            )
    
            st.success(f"✅ Data added! Total dataset: {len(st.session_state.df_purchase)} rows")
            st.rerun()
    
    # ===============================
    # TARGET & FEATURE
    # ===============================
    y = df["Revenue"]
    X = df.drop(columns=["Revenue"])
    
    if y.nunique() < 2:
        st.error("❌ Dataset only has 1 class, adding to another class.")
        st.stop()
    
    # ===============================
    # TRAIN MODEL
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    proba = model.predict_proba(X_test)
    
    # ===============================
    # 3️⃣ MODEL RESULT
    # ===============================
    st.header("3️⃣ Model Accuracy")
    
    st.success(f"🎯 Model Accuracy: {acc*100:.2f}%")
    
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
    
    # ===============================
    # 4️⃣ INPUT MANUAL TEST
    # ===============================
    st.header("4️⃣ Add sample data to predict")
    
    input_test = {}
    
    test_idx = 0
    for col in X.columns:
        if col in ["Month", "VisitorType", "Weekend"]:
            input_test[col] = st.selectbox(f"{col} (Test)", encoders[col].classes_, key=f"tab3_test_{test_idx}")
        else:
            input_test[col] = st.number_input(f"{col} (Test)", value=float(df[col].median()), key=f"tab3_test_{test_idx}")
        test_idx += 1
    
    # encode
    for col in ["Month", "VisitorType", "Weekend"]:
        input_test[col] = encoders[col].transform([input_test[col]])[0]
    
    input_pred = pd.DataFrame([input_test])
    
    # ===============================
    # 5️⃣ Prediction
    # ===============================
    if st.button("🔮 Predict Purchase", key="tab3_predict_btn"):
    
        pred = model.predict(input_pred)[0]
        prob = model.predict_proba(input_pred)[0][pred]
    
        if pred == 1:
            st.success(f"🛒 Result: Will Buy ( Confidence: {prob*100:.2f}%)")
        else:
            st.warning(f"❌ Result: Will not Buy ( Confidence: {prob*100:.2f}%)")
    
        # ===============================
        # 6️⃣ XAI
        # ===============================
        st.header("6️⃣ XAI: Dominant Factors")
    
        top_feats = fi_df.head(5)
    
        for _, row in top_feats.iterrows():
            st.write(f"• **{row['Feature']}** (importance: {row['Importance']:.4f})")
    
        # ===============================
        # 7️⃣ SHAP LOCAL
        # ===============================
        st.header("7️⃣ SHAP: Why this Product Sells Better?")
    
        explainer = shap.TreeExplainer(model)
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
