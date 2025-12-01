import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
from datetime import datetime
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from streamlit_lottie import st_lottie
import requests
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="EcoSort Edu - Smart Waste",
    page_icon="🌱",
    layout="wide",
)

# --- 2. CSS CUSTOM ---
st.markdown("""
<style>
    .stApp { background-color: #F1F8E9; }
    .stMarkdown, .stText, p, div, label, span, h1, h2, h3, h4, h5, h6 {
        color: #1b3a24 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 2px solid #C8E6C9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white !important;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover { background-color: #45a049; }
    .info-box {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #A5D6A7;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. DATABASE INFO SAMPAH ---
info_sampah = {
    "ORGANIK": {
        "judul": "🌱 SAMPAH ORGANIK",
        "definisi": "Sisa makhluk hidup yang mudah terurai oleh alam.",
        "sifat": "✅ Biodegradable (Mudah Membusuk)",
        "aksi": "Olah menjadi **Pupuk Kompos** atau **Pakan Ternak**.",
        "contoh": "Sisa makanan, kulit buah, sayur, daun kering."
    },
    "ANORGANIK": {
        "judul": "⚙️ SAMPAH ANORGANIK",
        "definisi": "Sampah buatan pabrik yang sangat sulit terurai.",
        "sifat": "❌ Non-Biodegradable (Tahan Ratusan Tahun)",
        "aksi": "Wajib **Daur Ulang (Recycle)** atau jual ke Bank Sampah.",
        "contoh": "Plastik, botol, kaleng, kaca, styrofoam."
    }
}

# --- 4. LOAD MODEL CNN ---
@st.cache_resource
def load_model():
    if not os.path.exists('model_sampah.h5'):
        return None
    return tf.keras.models.load_model('model_sampah.h5')

model = load_model()

# --- 5. FUNGSI PREDIKSI ---
def predict_image(img):
    if model is None:
        return "Model tidak ditemukan", 0.0
    img = img.resize((224,224))  # sesuaikan dengan input model
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    kelas = np.argmax(pred)
    prob = np.max(pred)
    label = "ORGANIK" if kelas == 0 else "ANORGANIK"
    return label, prob

# --- 6. NAVIGASI TAB ---
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Gambar", "📷 Kamera", "📊 Statistik", "📚 Edukasi"])

with tab1:
    st.header("Upload Gambar Sampah")
    uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diupload", use_column_width=True)
        label, prob = predict_image(img)
        st.success(f"✅ Prediksi: {label} ({prob*100:.2f}%)")

with tab2:
    st.header("Kamera / Live Stream")
    webrtc_streamer(key="kamera", mode=WebRtcMode.SENDRECV)

with tab3:
    st.header("Statistik Sampah")
    # Data dummy, bisa diganti dengan log hasil prediksi
    data = pd.DataFrame({
        "Kategori": ["Organik", "Anorganik"],
        "Jumlah": [120, 80]
    })
    fig = px.bar(data, x="Kategori", y="Jumlah", color="Kategori",
                 title="Distribusi Sampah Terdeteksi")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Edukasi Sampah")
    # Animasi Lottie opsional
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    lottie_recycle = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
    if lottie_recycle:
        st_lottie(lottie_recycle, height=200, key="recycle")

    for kategori, isi in info_sampah.items():
        st.markdown(f"""
        <div class="info-box">
            <h3>{isi['judul']}</h3>
            <p><b>Definisi:</b> {isi['definisi']}</p>
            <p><b>Sifat:</b> {isi['sifat']}</p>
            <p><b>Aksi:</b> {isi['aksi']}</p>
            <p><b>Contoh:</b> {isi['contoh']}</p>
        </div>
        """, unsafe_allow_html=True)
