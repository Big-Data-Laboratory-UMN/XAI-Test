import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ================================
# CONFIG
# ================================
DATASET_DIR = "ads_dataset"

st.set_page_config(page_title="Ad Design Effectiveness AI + XAI", layout="wide")
st.title("🎨 Ad Design Effectiveness AI + Explainability")

# ================================
# FEATURE EXTRACTION
# ================================
def analyze_visual_content(img: Image.Image):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    brightness = np.mean(gray)
    contrast = np.std(gray)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    mean_r = np.mean(img_cv[:, :, 2])
    mean_g = np.mean(img_cv[:, :, 1])
    mean_b = np.mean(img_cv[:, :, 0])

    # text area heuristic
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_area = 0
    for cnt in contours:
        x,y,wc,hc = cv2.boundingRect(cnt)
        area = wc * hc
        if area > 0.001 * (h*w):
            text_area += area

    text_ratio = text_area / (h*w)

    has_cta = 1 if text_ratio > 0.05 and contrast > 25 else 0

    return {
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge_density,
        "text_ratio": text_ratio,
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "has_cta": has_cta
    }

# ================================
# BUILD DATASET FROM FOLDER
# ================================
@st.cache_data
def build_dataset():
    data = []

    for label, cls in enumerate(["not_effective", "effective"]):
        folder = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg",".png",".jpeg")):
                path = os.path.join(folder, fname)
                try:
                    img = Image.open(path).convert("RGB")
                    feats = analyze_visual_content(img)
                    feats["label"] = label
                    data.append(feats)
                except:
                    print("Skip corrupted:", path)

    df = pd.DataFrame(data)
    return df

# ================================
# LOAD DATASET
# ================================
st.info("🔄 Building dataset from image folder...")
df = build_dataset()

if len(df) < 10:
    st.error("❌ Dataset terlalu kecil / tidak terbaca.")
    st.stop()

X = df.drop(columns=["label"])
y = df["label"]

# ================================
# TRAIN MODEL
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))

st.success(f"✅ Model trained. Validation Accuracy: {acc*100:.2f}%")

# ================================
# GLOBAL XAI
# ================================
st.header("🧠 Global XAI — What matters most?")

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

# ================================
# UPLOAD IMAGE
# ================================
st.header("🖼️ Test New Design")

uploaded = st.file_uploader("Upload ads image", type=["jpg","png","jpeg"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    feats = analyze_visual_content(img)

    st.subheader("🔎 Extracted Features")
    st.json(feats)

    input_df = pd.DataFrame([feats])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][pred]

    st.subheader("📢 Prediction Result")
    if pred == 1:
        st.success(f"✅ EFFECTIVE ({prob*100:.2f}%)")
    else:
        st.warning(f"⚠️ NOT EFFECTIVE ({prob*100:.2f}%)")

    # ================================
    # LOCAL XAI
    # ================================
    st.header("🔍 Local Explanation")

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

    st.subheader("🏆 Most Influential Factors:")
    for i, row in local_imp.head(3).iterrows():
        st.write(f"• **{row['Feature']}**")

