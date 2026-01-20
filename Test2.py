#!/usr/bin/env python
# coding: utf-8

# In[ ]:
def run_ad_design_app():

    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from PIL import Image, ImageEnhance
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    
    st.set_page_config(page_title="Ad Design Effectiveness AI + XAI", layout="wide")
    
    st.title("🎨 Visual Design Evaluation")
    st.header("\"Why Is This Design Considered More Effective?\"")
    
    
    # In[ ]:
    
    
    # ====================================
    # VISUAL ANALYZE
    # ====================================
    
    def analyze_visual_content(img: Image.Image):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
        h, w = gray.shape
    
        # ========== BASIC STATS ==========
        brightness = np.mean(gray)
        contrast = np.std(gray)
    
        mean_r = np.mean(img_cv[:, :, 2])
        mean_g = np.mean(img_cv[:, :, 1])
        mean_b = np.mean(img_cv[:, :, 0])
    
        # ========== TEXT AREA ESTIMATION ==========
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
    
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        text_area = 0
        large_blocks = []
    
        for cnt in contours:
            x,y,wc,hc = cv2.boundingRect(cnt)
            area = wc * hc
            if area > 0.001 * (h*w):  # threshold small noise
                text_area += area
                large_blocks.append((x,y,wc,hc))
    
        total_area = h * w
        text_ratio = text_area / total_area
    
        # ========== TEXT DENSITY CLASS ==========
        if text_ratio < 0.05:
            text_density = "Low"
        elif text_ratio < 0.15:
            text_density = "Medium"
        else:
            text_density = "High"
    
        # ========== ALIGNMENT CONSISTENCY ==========
        # Project text blocks to X axis
        if len(large_blocks) >= 2:
            xs = [x for x,y,wc,hc in large_blocks]
            alignment_std = np.std(xs)
            alignment = "Consistent" if alignment_std < w * 0.05 else "Inconsistent"
        else:
            alignment = "Consistent"
    
        # ========== CTA HEURISTIC ==========
        has_cta_detected = 0
        for (x,y,wc,hc) in large_blocks:
            if hc > h*0.1 and wc > w*0.3 and y > h*0.4:
                has_cta_detected = 1
    
        return {
            "brightness": brightness,
            "contrast": contrast,
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b": mean_b,
            "text_ratio": text_ratio,
            "text_density": text_density,
            "alignment": alignment,
            "has_cta_detected": has_cta_detected
        }
    
    
    # In[ ]:
    
    
    # ====================================
    # HELPER FUNCTIONS
    # ====================================
    
    def extract_features(img: Image.Image):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
        brightness = np.mean(gray)
        contrast = np.std(gray)
    
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
    
        return brightness, contrast, edge_density
    
    
    def augment_image(img):
        aug = {}
    
        # Brightness
        aug["Bright+"] = ImageEnhance.Brightness(img).enhance(1.5)
        aug["Bright-"] = ImageEnhance.Brightness(img).enhance(0.6)
    
        # Contrast
        aug["Contrast+"] = ImageEnhance.Contrast(img).enhance(1.5)
        aug["Contrast-"] = ImageEnhance.Contrast(img).enhance(0.6)
    
        # Flip
        aug["Flip"] = img.transpose(Image.FLIP_LEFT_RIGHT)
    
        return aug
    
    
    # In[ ]:
    
    
    # ====================================
    # SIMULATED TRAINING DATA
    # ====================================
    # Normally this comes from labeled dataset of ads
    
    np.random.seed(42)
    
    train_data = []
    for i in range(200):
        brightness = np.random.uniform(50, 200)
        contrast = np.random.uniform(10, 80)
        edge_density = np.random.uniform(0.01, 0.2)
        has_cta = np.random.randint(0, 2)
    
        # Fake rule to generate label
        score = (
            0.4 * (brightness > 120) +
            0.4 * (contrast > 30) +
            0.5 * has_cta +
            0.3 * (edge_density < 0.12)
        )
    
        effective = 1 if score >= 1.2 else 0
    
        train_data.append([
            brightness, contrast, edge_density, has_cta, effective
        ])
    
    df = pd.DataFrame(train_data, columns=[
        "brightness", "contrast", "edge_density", "has_cta", "effective"
    ])
    
    X = df.drop(columns=["effective"])
    y = df["effective"]
    
    
    # In[ ]:
    
    
    # ====================================
    # TRAIN MODEL
    # ====================================
    model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    model.fit(X, y)
    
    # ====================================
    # GLOBAL XAI
    # ====================================
    st.header("🧠 Global XAI — Elemen Visual Paling Berpengaruh")
    
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
    
    
    # In[ ]:
    
    
    # ====================================
    # UI INPUT
    # ====================================
    st.header("🖼️ Upload Desain Iklan")
    
    uploaded = st.file_uploader("Upload image iklan", type=["jpg", "png", "jpeg"])
    
    #has_cta = st.selectbox("Apakah ada CTA (Buy Now, Learn More, etc)?", [0, 1])
    #format_type = st.selectbox("Format Iklan", ["Square", "Vertical", "Horizontal"])
    
    #format_map = {
    #    "Square": 0,
    #    "Vertical": 1,
    #    "Horizontal": 2
    #}
    
    
    # In[ ]:
    
    
    # ====================================
    # MAIN LOGIC
    # ====================================
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
    
        st.image(img, caption="Original Design", width=300)
    
        # ===============================
        # ANALISIS VISUAL KONTEN
        # ===============================
        st.header("🔎 Analisis Otomatis Konten Visual")
    
        analysis = analyze_visual_content(img)
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.metric("Brightness", f"{analysis['brightness']:.1f}")
            st.metric("Contrast", f"{analysis['contrast']:.1f}")
            st.metric("Mean R", f"{analysis['mean_r']:.1f}")
            st.metric("Mean G", f"{analysis['mean_g']:.1f}")
            st.metric("Mean B", f"{analysis['mean_b']:.1f}")
    
        with col2:
            st.metric("Text / Image Ratio", f"{analysis['text_ratio']*100:.2f}%")
            st.metric("Text Density", analysis["text_density"])
            st.metric("Alignment", analysis["alignment"])
            st.metric("CTA Detected (AI)", "YES" if analysis["has_cta_detected"] else "NO")
    
        st.markdown("""
        **Interpretasi Singkat:**
        - Text Density tinggi → risiko clutter
        - Kontras rendah → iklan sulit dibaca
        - Alignment tidak konsisten → visual tidak rapi
        - CTA terdeteksi → potensi konversi lebih tinggi
        """)
    
        st.image(img, caption="Original Design", width=300)
    
        brightness, contrast, edge_density = extract_features(img)
        has_cta = 1 if analysis["has_cta_detected"] else 0
        
        input_df = pd.DataFrame([{
            "brightness": brightness,
            "contrast": contrast,
            "edge_density": edge_density,
            "has_cta": has_cta
        }])
    
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][pred]
    
        st.subheader("📢 Hasil Prediksi")
    
        if pred == 1:
            st.success(f"✅ Prediksi: EFFECTIVE ({prob*100:.2f}%)")
        else:
            st.warning(f"⚠️ Prediksi: LESS EFFECTIVE ({prob*100:.2f}%)")
    
    
    # In[ ]:
    
    
    # ====================================
        # LOCAL XAI
        # ====================================
        st.header("🔍 Kenapa hasilnya seperti ini? (Local XAI)")
    
        perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    
        local_imp = pd.DataFrame({
            "Feature": X.columns,
            "Importance": perm.importances_mean
        }).sort_values(by="Importance", ascending=False)
    
        fig2, ax2 = plt.subplots()
        ax2.barh(local_imp["Feature"], local_imp["Importance"])
        ax2.invert_yaxis()
        ax2.set_title("Local Explanation (Permutation Importance)")
        st.pyplot(fig2)
    
        st.dataframe(local_imp)
    
        st.subheader("🏆 Faktor Paling Menentukan:")
        for i, row in local_imp.head(3).iterrows():
            st.write(f"• **{row['Feature']}**")
    
    
    # In[ ]:
    
    
    # ====================================
        # AUGMENTATION
        # ====================================
        st.header("🧪 Simulasi Augmentasi Desain + Interpretasi")
    
        aug_imgs = augment_image(img)
    
        cols = st.columns(len(aug_imgs))
    
        for i, (name, aug_img) in enumerate(aug_imgs.items()):
            with cols[i]:
                st.image(aug_img, caption=name, width=200)
    
                b, c, e = extract_features(aug_img)
    
                aug_input = pd.DataFrame([{
                    "brightness": b,
                    "contrast": c,
                    "edge_density": e,
                    "has_cta": has_cta
                }])
    
                p = model.predict(aug_input)[0]
                pr = model.predict_proba(aug_input)[0][p]
    
                if p == 1:
                    st.success(f"Effective ({pr*100:.1f}%)")
                else:
                    st.warning(f"Less Effective ({pr*100:.1f}%)")

