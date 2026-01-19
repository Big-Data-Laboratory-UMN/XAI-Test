#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageEnhance

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Ad Design Effectiveness AI + XAI", layout="wide")

st.title("🎨 AI Evaluasi Efektivitas Desain Iklan Digital + XAI")


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
    format_type = np.random.randint(0, 3)  # 0 square, 1 vertical, 2 horizontal

    # Fake rule to generate label
    score = (
        0.4 * (brightness > 120) +
        0.4 * (contrast > 30) +
        0.5 * has_cta +
        0.3 * (edge_density < 0.12)
    )

    effective = 1 if score >= 1.2 else 0

    train_data.append([
        brightness, contrast, edge_density, has_cta, format_type, effective
    ])

df = pd.DataFrame(train_data, columns=[
    "brightness", "contrast", "edge_density", "has_cta", "format_type", "effective"
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

has_cta = st.selectbox("Apakah ada CTA (Buy Now, Learn More, etc)?", [0, 1])
format_type = st.selectbox("Format Iklan", ["Square", "Vertical", "Horizontal"])

format_map = {
    "Square": 0,
    "Vertical": 1,
    "Horizontal": 2
}


# In[ ]:


# ====================================
# MAIN LOGIC
# ====================================
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")

    st.image(img, caption="Original Design", width=300)

    brightness, contrast, edge_density = extract_features(img)

    input_df = pd.DataFrame([{
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge_density,
        "has_cta": has_cta,
        "format_type": format_map[format_type]
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
                "has_cta": has_cta,
                "format_type": format_map[format_type]
            }])

            p = model.predict(aug_input)[0]
            pr = model.predict_proba(aug_input)[0][p]

            if p == 1:
                st.success(f"Effective ({pr*100:.1f}%)")
            else:
                st.warning(f"Less Effective ({pr*100:.1f}%)")


# In[ ]:


# ====================================
# ETHICS
# ====================================
st.header("⚖️ Diskusi Etika")

st.markdown("""
**Batas antara optimasi desain dan manipulasi visual:**

- AI sebaiknya:
  - Membantu meningkatkan *clarity, readability, usability*
  - Membantu brand menyampaikan pesan lebih efektif

- AI tidak boleh:
  - Mengeksploitasi bias psikologis secara berlebihan
  - Memanipulasi emosi (fear, addiction, FOMO berlebihan)
  - Menyesatkan secara visual

**Prinsip etis:**
> AI = alat bantu desain, bukan mesin manipulasi perilaku manusia.
""")

