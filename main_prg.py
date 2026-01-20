#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from modules.Test1 import run_media_engagement_app
from modules.Test2 import run_ad_design_app
from modules.Test3 import run_purchase_app

st.set_page_config(page_title="AI Marketing Suite + XAI", layout="wide")

st.title("🚀 XAI Suite")

tab1, tab2, tab3 = st.tabs([
    "📊 Viralitas Konten",
    "🎨 Evaluasi Desain Iklan",
    "🛒 Prediksi Pembelian"
])

with tab1:
    run_media_engagement_app()

with tab2:
    run_ad_design_app()

with tab3:
    run_purchase_app()

