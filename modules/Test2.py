#!/usr/bin/env python
# coding: utf-8

def run_ad_design_app():

    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from PIL import Image, ImageEnhance
    
    from tensorflow.keras.models import load_model
    
    st.set_page_config(page_title="Ad Design Effectiveness AI + XAI", layout="wide")
    
    st.title("🎨 Visual Design Evaluation")
    st.header("\"Why Is This Design Considered More Effective?\"")
    
    # ====================================
    # LOAD CNN MODEL
    # ====================================
    @st.cache_resource
    def load_cnn():
        return load_model("ads_cnn_model.h5")

    model = load_cnn()
    
    # ====================================
    # VISUAL ANALYZE
    # ====================================
    def analyze_visual_content(img: Image.Image):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
        h, w = gray.shape
    
        brightness = np.mean(gray)
        contrast = np.std(gray)
    
        mean_r = np.mean(img_cv[:, :, 2])
        mean_g = np.mean(img_cv[:, :, 1])
        mean_b = np.mean(img_cv[:, :, 0])
    
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
    
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        text_area = 0
        large_blocks = []
    
        for cnt in contours:
            x,y,wc,hc = cv2.boundingRect(cnt)
            area = wc * hc
            if area > 0.001 * (h*w):
                text_area += area
                large_blocks.append((x,y,wc,hc))
    
        total_area = h * w
        text_ratio = text_area / total_area
    
        if text_ratio < 0.05:
            text_density = "Low"
        elif text_ratio < 0.15:
            text_density = "Medium"
        else:
            text_density = "High"
    
        if len(large_blocks) >= 2:
            xs = [x for x,y,wc,hc in large_blocks]
            alignment_std = np.std(xs)
            alignment = "Consistent" if alignment_std < w * 0.05 else "Inconsistent"
        else:
            alignment = "Consistent"
    
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
    
    # ====================================
    # AUGMENTATION
    # ====================================
    def augment_image(img):
        aug = {}
        aug["Bright+"] = ImageEnhance.Brightness(img).enhance(1.5)
        aug["Bright-"] = ImageEnhance.Brightness(img).enhance(0.6)
        aug["Contrast+"] = ImageEnhance.Contrast(img).enhance(1.5)
        aug["Contrast-"] = ImageEnhance.Contrast(img).enhance(0.6)
        aug["Flip"] = img.transpose(Image.FLIP_LEFT_RIGHT)
        return aug
    
    # ====================================
    # UI INPUT
    # ====================================
    st.header("🖼️ Upload Image File")
    uploaded = st.file_uploader("Upload ads image", type=["jpg", "png", "jpeg"])
    
    # ====================================
    # MAIN LOGIC
    # ====================================
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Original Design", width=300)
    
        # ===============================
        # ANALISIS VISUAL
        # ===============================
        st.header("🔎 Analysis on Ads")
    
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
    
        # ===============================
        # CNN PREDICTION
        # ===============================
        img_resized = img.resize((224,224))
        img_arr = np.array(img_resized) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)
    
        prob = model.predict(img_arr)[0][0]
        pred = 1 if prob > 0.5 else 0
    
        st.subheader("📢 Prediction Results")
    
        if pred == 1:
            st.success(f"✅ Prediction: EFFECTIVE ({prob*100:.2f}%)")
        else:
            st.warning(f"⚠️ Prediction: LESS EFFECTIVE ({(1-prob)*100:.2f}%)")
    
        # ===============================
        # AUGMENTATION SIMULATION
        # ===============================
        st.header("🧪 Augmentation Simulation")
    
        aug_imgs = augment_image(img)
        cols = st.columns(len(aug_imgs))
    
        for i, (name, aug_img) in enumerate(aug_imgs.items()):
            with cols[i]:
                st.image(aug_img, caption=name, width=200)
    
                aug_resized = aug_img.resize((224,224))
                aug_arr = np.array(aug_resized) / 255.0
                aug_arr = np.expand_dims(aug_arr, axis=0)
    
                pr = model.predict(aug_arr)[0][0]
                p = 1 if pr > 0.5 else 0
    
                if p == 1:
                    st.success(f"Effective ({pr*100:.1f}%)")
                else:
                    st.warning(f"Less Effective ({(1-pr)*100:.1f}%)")
                    
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
