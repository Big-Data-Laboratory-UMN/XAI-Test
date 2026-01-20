#!/usr/bin/env python
# coding: utf-8

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
    
    
    # ====================================
    # VISUAL ANALYZE (IMPROVED)
    # ====================================
    
    def analyze_visual_content(img: Image.Image):
        import pytesseract
        
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
        h, w = gray.shape
    
        # ========== BASIC STATS ==========
        brightness = np.mean(gray)
        contrast = np.std(gray)
    
        mean_r = np.mean(img_cv[:, :, 2])
        mean_g = np.mean(img_cv[:, :, 1])
        mean_b = np.mean(img_cv[:, :, 0])
    
        # ========== COLOR SATURATION (NEW) ==========
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1]) / 255.0  # Normalize to 0-1
        
        # ========== NUMBER OF DOMINANT COLORS (NEW) ==========
        # Reshape image to list of pixels
        pixels = img_cv.reshape(-1, 3)
        # Use k-means to find dominant colors
        from sklearn.cluster import KMeans
        n_clusters = min(10, len(pixels))  # Max 10 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Count significant colors (those that appear in >5% of image)
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        total_pixels = len(pixels)
        significant_colors = sum(1 for count in counts if count / total_pixels > 0.05)
        num_colors = significant_colors
        
        # ========== FACE DETECTION (NEW) ==========
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        has_faces = 1 if len(faces) > 0 else 0
    
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
    
        # ========== CTA DETECTION WITH OCR (IMPROVED) ==========
        # Extract text from image
        try:
            text = pytesseract.image_to_string(img, config='--psm 6')
            text = text.lower()
            
            # Common CTA phrases
            cta_keywords = [
                'buy now', 'shop now', 'get started', 'sign up', 'subscribe',
                'learn more', 'download', 'register', 'join', 'order now',
                'get yours', 'claim', 'try free', 'book now', 'reserve',
                'apply now', 'contact us', 'get quote', 'start free trial',
                'add to cart', 'checkout', 'purchase', 'enroll', 'watch now',
                'click here', 'discover', 'explore', 'find out', 'see more'
            ]
            
            # Check if any CTA keyword is in the text
            has_cta_detected = 0
            detected_ctas = []
            for keyword in cta_keywords:
                if keyword in text:
                    has_cta_detected = 1
                    detected_ctas.append(keyword)
            
            # Fallback to geometric heuristic if no text CTA found
            if has_cta_detected == 0:
                for (x,y,wc,hc) in large_blocks:
                    if hc > h*0.1 and wc > w*0.3 and y > h*0.4:
                        has_cta_detected = 1
                        detected_ctas = ["(geometric detection)"]
                        break
        except:
            # If OCR fails, use geometric heuristic
            has_cta_detected = 0
            detected_ctas = []
            for (x,y,wc,hc) in large_blocks:
                if hc > h*0.1 and wc > w*0.3 and y > h*0.4:
                    has_cta_detected = 1
                    detected_ctas = ["(geometric detection)"]
                    break
    
        # ========== EDGE DENSITY (for visual complexity) ==========
        edge_density = np.sum(edges > 0) / edges.size
    
        return {
            "brightness": brightness,
            "contrast": contrast,
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b": mean_b,
            "color_saturation": saturation,
            "num_colors": num_colors,
            "has_faces": has_faces,
            "text_ratio": text_ratio,
            "text_density": text_density,
            "alignment": alignment,
            "has_cta_detected": has_cta_detected,
            "detected_ctas": detected_ctas,
            "edge_density": edge_density
        }
    
    
    # ====================================
    # HELPER FUNCTIONS (IMPROVED)
    # ====================================
    
    def extract_features(img: Image.Image):
        analysis = analyze_visual_content(img)
        return (
            analysis["brightness"],
            analysis["contrast"],
            analysis["edge_density"],
            analysis["color_saturation"],
            analysis["num_colors"],
            analysis["has_faces"]
        )
    
    
    def augment_image(img):
        aug = {}
    
        # Brightness
        aug["Bright+"] = ImageEnhance.Brightness(img).enhance(1.5)
        aug["Bright-"] = ImageEnhance.Brightness(img).enhance(0.6)
    
        # Contrast
        aug["Contrast+"] = ImageEnhance.Contrast(img).enhance(1.5)
        aug["Contrast-"] = ImageEnhance.Contrast(img).enhance(0.6)
        
        # Saturation
        aug["Saturation+"] = ImageEnhance.Color(img).enhance(1.8)
        aug["Saturation-"] = ImageEnhance.Color(img).enhance(0.5)
    
        # Flip
        aug["Flip"] = img.transpose(Image.FLIP_LEFT_RIGHT)
    
        return aug
    
    
    # ====================================
    # SIMULATED TRAINING DATA (IMPROVED RULES)
    # ====================================
    
    np.random.seed(42)
    
    train_data = []
    for i in range(200):
        brightness = np.random.uniform(50, 200)
        contrast = np.random.uniform(10, 80)
        edge_density = np.random.uniform(0.01, 0.2)
        has_cta = np.random.randint(0, 2)
        color_saturation = np.random.uniform(0.2, 1.0)
        num_colors = np.random.randint(1, 8)
        has_faces = np.random.randint(0, 2)
    
        # IMPROVED RESEARCH-BASED RULE
        score = (
            0.35 * (50 <= brightness <= 180) +          # Moderate brightness
            0.30 * (contrast > 40) +                     # High contrast
            0.40 * has_cta +                             # CTA presence
            0.25 * (0.05 < edge_density < 0.10) +       # Moderate complexity
            0.30 * (color_saturation > 0.5) +           # High saturation
            0.20 * (num_colors <= 3) +                   # Limited palette
            0.15 * has_faces                             # Human faces
        )
    
        effective = 1 if score >= 1.2 else 0
    
        train_data.append([
            brightness, contrast, edge_density, has_cta,
            color_saturation, num_colors, has_faces, effective
        ])
    
    df = pd.DataFrame(train_data, columns=[
        "brightness", "contrast", "edge_density", "has_cta",
        "color_saturation", "num_colors", "has_faces", "effective"
    ])
    
    X = df.drop(columns=["effective"])
    y = df["effective"]
    
    
    # ====================================
    # TRAIN MODEL
    # ====================================
    model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    model.fit(X, y)
    
    # ====================================
    # GLOBAL XAI
    # ====================================
    st.header("🧠 Global XAI — Most Influential Visual Elements")
    
    fi = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots()
    ax.barh(fi["Feature"], fi["Importance"], color='steelblue')
    ax.invert_yaxis()
    ax.set_title("Global Feature Importance")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)
    
    st.dataframe(fi)
    
    
    # ====================================
    # UI INPUT
    # ====================================
    st.header("🖼️ Upload Ad Design")
    
    uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    
    
    # ====================================
    # MAIN LOGIC
    # ====================================
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
    
        st.image(img, caption="Original Design", width=300)
    
        # ===============================
        # VISUAL CONTENT ANALYSIS
        # ===============================
        st.header("🔎 Visual Content Analysis")
    
        analysis = analyze_visual_content(img)
    
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.metric("Brightness", f"{analysis['brightness']:.1f}")
            st.metric("Contrast", f"{analysis['contrast']:.1f}")
            st.metric("Edge Density", f"{analysis['edge_density']:.3f}")
    
        with col2:
            st.metric("Color Saturation", f"{analysis['color_saturation']:.2f}")
            st.metric("Number of Colors", f"{analysis['num_colors']}")
            st.metric("Has Faces", "YES" if analysis["has_faces"] else "NO")
    
        with col3:
            st.metric("Text Ratio", f"{analysis['text_ratio']*100:.2f}%")
            st.metric("Text Density", analysis["text_density"])
            cta_display = "YES" if analysis["has_cta_detected"] else "NO"
            st.metric("CTA Detected", cta_display)
            if analysis["has_cta_detected"] and analysis["detected_ctas"]:
                st.caption(f"Found: {', '.join(analysis['detected_ctas'][:3])}")  # Show up to 3 CTAs
    
        # ====================================
        # EFFECTIVENESS INDICATORS
        # ====================================
        st.subheader("📊 Effectiveness Indicators")
        
        indicators = []
        
        # Brightness check
        if 50 <= analysis['brightness'] <= 180:
            indicators.append(("✅", "Optimal Brightness", "Moderate brightness improves visibility"))
        else:
            indicators.append(("⚠️", "Suboptimal Brightness", "Too dark or too bright reduces effectiveness"))
        
        # Contrast check
        if analysis['contrast'] > 40:
            indicators.append(("✅", "High Contrast", "Good contrast improves readability"))
        else:
            indicators.append(("⚠️", "Low Contrast", "Increase contrast for better visibility"))
        
        # Saturation check
        if analysis['color_saturation'] > 0.5:
            indicators.append(("✅", "High Color Saturation", "Vibrant colors are more engaging"))
        else:
            indicators.append(("⚠️", "Low Color Saturation", "Consider more vibrant colors"))
        
        # Color count check
        if analysis['num_colors'] <= 3:
            indicators.append(("✅", "Simple Color Palette", "Limited colors improve clarity"))
        else:
            indicators.append(("⚠️", "Complex Color Palette", "Too many colors may cause clutter"))
        
        # Complexity check
        if 0.05 < analysis['edge_density'] < 0.10:
            indicators.append(("✅", "Optimal Visual Complexity", "Balanced detail level"))
        else:
            indicators.append(("⚠️", "Suboptimal Complexity", "Too simple or too cluttered"))
        
        # CTA check
        if analysis['has_cta_detected']:
            indicators.append(("✅", "CTA Detected", "Call-to-action increases effectiveness"))
        else:
            indicators.append(("⚠️", "No CTA Detected", "Consider adding a clear call-to-action"))
        
        # Face check
        if analysis['has_faces']:
            indicators.append(("✅", "Human Face Detected", "Faces increase engagement"))
        
        for emoji, title, desc in indicators:
            st.write(f"{emoji} **{title}**: {desc}")
    
        # ====================================
        # PREDICTION
        # ====================================
        brightness, contrast, edge_density, color_saturation, num_colors, has_faces = extract_features(img)
        has_cta = 1 if analysis["has_cta_detected"] else 0
        
        input_df = pd.DataFrame([{
            "brightness": brightness,
            "contrast": contrast,
            "edge_density": edge_density,
            "has_cta": has_cta,
            "color_saturation": color_saturation,
            "num_colors": num_colors,
            "has_faces": has_faces
        }])
    
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][pred]
        
        st.subheader("📢 Prediction Result")
    
        if pred == 1:
            st.success(f"✅ Prediction: EFFECTIVE ({prob*100:.2f}%)")
        else:
            st.warning(f"⚠️ Prediction: LESS EFFECTIVE ({prob*100:.2f}%)")
    
        # ====================================
        # AUGMENTATION
        # ====================================
        st.header("🧪 Augmentation Simulation")
    
        aug_imgs = augment_image(img)
    
        cols = st.columns(len(aug_imgs))
    
        for i, (name, aug_img) in enumerate(aug_imgs.items()):
            with cols[i]:
                st.image(aug_img, caption=name, width=150)
    
                b, c, e, cs, nc, hf = extract_features(aug_img)
    
                aug_input = pd.DataFrame([{
                    "brightness": b,
                    "contrast": c,
                    "edge_density": e,
                    "has_cta": has_cta,
                    "color_saturation": cs,
                    "num_colors": nc,
                    "has_faces": hf
                }])
    
                p = model.predict(aug_input)[0]
                pr = model.predict_proba(aug_input)[0][p]
    
                if p == 1:
                    st.success(f"Effective ({pr*100:.1f}%)")
                else:
                    st.warning(f"Less Effective ({pr*100:.1f}%)")
    
        # ====================================
        # LOCAL XAI
        # ====================================
        st.header("🔍 Why This Result? (Local XAI)")
    
        perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    
        local_imp = pd.DataFrame({
            "Feature": X.columns,
            "Importance": perm.importances_mean
        }).sort_values(by="Importance", ascending=False)
    
        fig2, ax2 = plt.subplots()
        ax2.barh(local_imp["Feature"], local_imp["Importance"], color='coral')
        ax2.invert_yaxis()
        ax2.set_title("Local Explanation (Permutation Importance)")
        ax2.set_xlabel("Importance Score")
        st.pyplot(fig2)
    
        st.dataframe(local_imp)
    
        st.subheader("🏆 Most Determining Factors:")
        for i, row in local_imp.head(3).iterrows():
            st.write(f"• **{row['Feature']}** (Importance: {row['Importance']:.4f})")

if __name__ == "__main__":
    run_ad_design_app()
