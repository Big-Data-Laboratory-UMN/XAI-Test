#!/usr/bin/env python
# coding: utf-8

def run_media_engagement_app():
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
    st.title("📊 Social Media Content Engagement Prediction (High vs Low Engagement)")
    st.header("\"Why does This content Get More Likes and Shares?\"")
    
    # ===============================
    # SESSION STATE
    # ===============================
    if "df" not in st.session_state:
        st.session_state.df = None
    if "model" not in st.session_state:
        st.session_state.model = None
    if "encoders" not in st.session_state:
        st.session_state.encoders = None
    if "trained" not in st.session_state:
        st.session_state.trained = False
    
    # ===============================
    # CONFIG
    # ===============================
    FEATURES = [
        "Type", "Category", "Post Month", "Post Weekday", "Post Hour", "Paid",
        "Lifetime Post Total Reach", "Lifetime Engaged Users", "comment", "like", "share"
    ]
    
    # ===============================
    # FUNCTION: TRAIN PIPELINE
    # ===============================
    def train_model(df):
    
        df = df.copy().dropna()
    
        encoders = {}
        for col in ["Type"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    
        if "Engagement_Label" not in df.columns:
            median_interaction = df["Total Interactions"].median()
            df["Engagement_Label"] = (df["Total Interactions"] > median_interaction).astype(int)
    
        X = df[FEATURES]
        y = df["Engagement_Label"]
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
    
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
    
        return model, encoders, acc, X_train, y_train
    
    # ===============================
    # 1️⃣ UPLOAD CSV
    # ===============================
    st.header("1️⃣ Upload Dataset CSV")
    
    # ADDED UNIQUE KEY FOR TAB 1
    uploaded = st.file_uploader("Upload CSV (next upload will be appended)", type=["csv"], key="tab1_media_engagement_uploader")
    
    if st.button("📥 Upload & Train Model", key="tab1_upload_train_btn"):
        if uploaded is not None:
            new_df = pd.read_csv(uploaded, sep=';')
    
            if st.session_state.df is None:
                st.session_state.df = new_df
            else:
                st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)
    
            model, encoders, acc, X_train, y_train = train_model(st.session_state.df)
    
            st.session_state.model = model
            st.session_state.encoders = encoders
            st.session_state.acc = acc
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.trained = True
    
            st.success("✅ Dataset updated, retrain model")
    
    # ===============================
    # 2️⃣ INPUT MANUAL ROW
    # ===============================
    st.header("2️⃣ Add new Data to Dataset")
    
    if st.session_state.encoders is None:
        type_options = ["Photo", "Status"]
    else:
        type_options = st.session_state.encoders["Type"].classes_
    
    type_input = st.selectbox("Type", type_options, key="tab1_type_select")
    category = st.number_input("Category", 1, 3, 2, key="tab1_category")
    post_month = st.number_input("Post Month", 1, 12, 12, key="tab1_month")
    weekday = st.number_input("Post Weekday (1=Mon, 7=Sun)", 1, 7, 3, key="tab1_weekday")
    hour = st.number_input("Post Hour", 0, 23, 10, key="tab1_hour")
    paid = st.selectbox("Paid Promotion?", [0, 1], key="tab1_paid")
    reach = st.number_input("Lifetime Post Total Reach", 0, 1000000, 10000, key="tab1_reach")
    engaged = st.number_input("Lifetime Engaged Users", 0, 100000, 500, key="tab1_engaged")
    comment = st.number_input("Comment", 0, 5000, 10, key="tab1_comment")
    like = st.number_input("Like", 0, 100000, 100, key="tab1_like")
    share = st.number_input("Share", 0, 5000, 10, key="tab1_share")
    
    if st.button("➕ Add & Re-train", key="tab1_add_retrain_btn"):
    
        if st.session_state.df is None:
            st.warning("⚠️ Upload data to begin")
        else:
            if st.session_state.encoders:
                type_encoded = st.session_state.encoders["Type"].transform([type_input])[0]
            else:
                type_encoded = 0
    
            new_row = {
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
                "share": share,
                "Total Interactions": comment + like + share
            }
    
            st.session_state.df = pd.concat(
                [st.session_state.df, pd.DataFrame([new_row])],
                ignore_index=True
            )
    
            model, encoders, acc, X_train, y_train = train_model(st.session_state.df)
    
            st.session_state.model = model
            st.session_state.encoders = encoders
            st.session_state.acc = acc
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.trained = True
    
            st.success("✅ Data Added, retrain model")
    
    # ===============================
    # 3️⃣ EVALUATION + GLOBAL XAI
    # ===============================
    if st.session_state.trained:
    
        st.header("3️⃣ Model Evaluation & Global Feature Importance")
    
        st.metric("🎯 Accuracy", f"{st.session_state.acc*100:.2f}%")
    
        fi = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": st.session_state.model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
    
        fig, ax = plt.subplots()
        ax.barh(fi["Feature"], fi["Importance"])
        ax.invert_yaxis()
        ax.set_title("Global Feature Importance")
        st.pyplot(fig)
        plt.close(fig)
    
        st.dataframe(fi)
    
    # ===============================
    # 4️⃣ INPUT UNTUK TEST PREDIKSI
    # ===============================
    if st.session_state.trained:
    
        st.header("4️⃣ Input data for Predictions")
    
        type_input_p = st.selectbox("Type (prediksi)", type_options, key="tab1_p1")
        category_p = st.number_input("Category (prediksi)", 1, 3, 2, key="tab1_p2")
        post_month_p = st.number_input("Post Month (prediksi)", 1, 12, 12, key="tab1_p3")
        weekday_p = st.number_input("Post Weekday (prediksi)", 1, 7, 3, key="tab1_p4")
        hour_p = st.number_input("Post Hour (prediksi)", 0, 23, 10, key="tab1_p5")
        paid_p = st.selectbox("Paid Promotion? (prediksi)", [0, 1], key="tab1_p6")
        reach_p = st.number_input("Lifetime Post Total Reach (prediksi)", 0, 1000000, 10000, key="tab1_p7")
        engaged_p = st.number_input("Lifetime Engaged Users (prediksi)", 0, 100000, 500, key="tab1_p8")
        comment_p = st.number_input("Comment (prediksi)", 0, 5000, 10, key="tab1_p9")
        like_p = st.number_input("Like (prediksi)", 0, 100000, 100, key="tab1_p10")
        share_p = st.number_input("Share (prediksi)", 0, 5000, 10, key="tab1_p11")
    
        if st.button("🔮 Predict", key="tab1_predict_btn"):
    
            type_encoded_p = st.session_state.encoders["Type"].transform([type_input_p])[0]
    
            input_pred = pd.DataFrame([{
                "Type": type_encoded_p,
                "Category": category_p,
                "Post Month": post_month_p,
                "Post Weekday": weekday_p,
                "Post Hour": hour_p,
                "Paid": paid_p,
                "Lifetime Post Total Reach": reach_p,
                "Lifetime Engaged Users": engaged_p,
                "comment": comment_p,
                "like": like_p,
                "share": share_p
            }])
    
            model = st.session_state.model
    
            pred = model.predict(input_pred)[0]
            prob = model.predict_proba(input_pred)[0][pred]
    
            if pred == 1:
                st.success(f"🔥 Prediksi: HIGH Engagement ({prob*100:.2f}%)")
            else:
                st.warning(f"❄️ Prediksi: LOW Engagement ({prob*100:.2f}%)")
    
            # ===============================
            # 5️⃣ XAI SIMPLE
            # ===============================
            st.header("5️⃣ XAI — Dominant Factors (Global-based)")
    
            contrib = pd.DataFrame({
                "Feature": FEATURES,
                "Value": input_pred.iloc[0].values,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
    
            st.dataframe(contrib)
    
            # ===============================
            # 6️⃣ SHAP — Kenapa Konten Ini Viral / Tidak
            # ===============================
            st.header("6️⃣ SHAP — Why The Content Go Viral?")
            
            explainer = shap.TreeExplainer(model)
            
            # SHAP explanation object (new API)
            shap_exp = explainer(input_pred)
            
            # predicted class
            pred_class = int(pred)
            
            # shap values for that class
            shap_vals = shap_exp.values[0][:, pred_class]
            
            base_val = shap_exp.base_values[0][pred_class]
            
            # build explanation object for single class
            shap_explanation = shap.Explanation(
                values=shap_vals,
                base_values=base_val,
                data=input_pred.iloc[0],
                feature_names=input_pred.columns
            )
            
            fig_local, ax_local = plt.subplots()
            shap.plots.waterfall(shap_explanation, show=False)
            st.pyplot(fig_local)
            plt.close(fig_local)
