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

# --- 2. CSS BERSIH & JELAS (VERSI TERBAIK) ---
st.markdown("""
<style>
    /* Background Mint Cream yang kamu suka */
    .stApp {
        background-color: #F1F8E9;
    }
    
    /* Memaksa Teks jadi Gelap (Agar terbaca di mode apapun) */
    .stMarkdown, .stText, p, div, label, span, h1, h2, h3, h4, h5, h6 {
        color: #1b3a24 !important;
    }
    
    /* Sidebar Putih */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 2px solid #C8E6C9;
    }

    /* Tombol Hijau Segar */
    .stButton>button {
        background-color: #4CAF50;
        color: white !important;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }

    /* Kotak Info Edukasi */
    .info-box {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #A5D6A7;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. DATABASE & ASET ---
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

@st.cache_resource
def load_model():
    if not os.path.exists('model_sampah.h5'): return None
    return tf.keras.models.load_model('model_sampah.h5')

@st.cache_data
def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

model = load_model()
lottie_recycle = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_tutvdkg0.json")

if 'history_data' not in st.session_state: st.session_state['history_data'] = []

# --- 4. ENGINE AI ---
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if model is not None:
            img_input = cv2.resize(img, (64, 64))
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            img_input = img_input.astype("float32") / 255.0
            img_input = np.expand_dims(img_input, axis=0)

            prediction = model.predict(img_input)
            confidence = prediction[0][0]

            if confidence > 0.5:
                label, color = "ORGANIK", (0, 200, 0)
                prob = confidence
            else:
                label, color = "ANORGANIK", (0, 100, 255)
                prob = 1 - confidence

            cv2.rectangle(img, (0, 0), (250, 60), (255, 255, 255), -1) 
            cv2.putText(img, f"{label} ({prob*100:.0f}%)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            h, w, _ = img.shape
            cv2.rectangle(img, (w//2-100, h//2-100), (w//2+100, h//2+100), color, 3)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def prediksi_gambar_diam(image):
    img_resized = image.resize((64, 64))
    img_array = np.array(img_resized)
    if img_array.shape == (64, 64): img_array = np.stack((img_array,)*3, axis=-1)
    if img_array.shape[-1] == 4: img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0) / 255.0 
    prediction = model.predict(img_array)
    skor = prediction[0][0]
    return ("ORGANIK", skor) if skor > 0.5 else ("ANORGANIK", 1 - skor)

def tampilkan_kartu_edukasi(label):
    data = info_sampah[label]
    st.markdown(f"""
    <div class="info-box">
        <h3 style="margin-top:0;">{data['judul']}</h3>
        <p style="font-size: 1.1em;"><b>{data['sifat']}</b></p>
        <p>{data['definisi']}</p>
        <hr>
        <p>💡 <b>Saran Pengelolaan:</b> {data['aksi']}</p>
    </div>
    """, unsafe_allow_html=True)

# ================= MAIN APP =================

# SIDEBAR
with st.sidebar:
    if lottie_recycle: st_lottie(lottie_recycle, height=150, key="anim_sidebar")
    st.title("EcoSort Edu 🌿")
    st.write("Aplikasi pintar untuk masa depan bumi yang lebih hijau.")
    st.divider()
    st.info("💡 **Tips:** Pisahkan tutup botol plastik sebelum dibuang!")
    st.caption("© 2025 Project UAS [Nama Kamu]")

if model is None:
    st.error("🚨 Model AI belum ada. Pastikan file 'model_sampah.h5' sudah di-upload ke GitHub!")
    st.stop()

# JUDUL UTAMA
st.title("🌍 Klasifikasi & Edukasi Sampah")
st.write("Selamat datang! Mari belajar memilah sampah dengan kecerdasan buatan.")

# TABS
tab_scan, tab_dash, tab_info = st.tabs(["📸 AI Scanner", "📊 Data Log", "ℹ️ Tentang"])

# === TAB 1: SCANNER ===
with tab_scan:
    st.subheader("🔍 Pilih Metode Deteksi:")
    mode = st.radio("", ["📁 Upload File", "📸 Kamera Foto", "📹 Video Live"], horizontal=True)
    st.write("---")

    if mode == "📁 Upload File":
        c1, c2 = st.columns([1, 1.2])
        with c1:
            up = st.file_uploader("Upload gambar sampah...", type=['jpg','png','jpeg'])
            if up:
                img = Image.open(up)
                st.image(img, caption="Gambar Input", use_container_width=True)
        with c2:
            if up:
                with st.spinner("Menganalisis..."):
                    lbl, conf = prediksi_gambar_diam(img)
                    tampilkan_kartu_edukasi(lbl)
                    st.metric("Tingkat Keyakinan AI", f"{conf*100:.1f}%")
                    if st.button("💾 Simpan ke Data Log"):
                        st.session_state['history_data'].append({"Waktu": datetime.now().strftime("%H:%M"), "Jenis": lbl})
                        st.toast("Data berhasil disimpan!", icon="✅")

    elif mode == "📸 Kamera Foto":
        c1, c2 = st.columns(2)
        with c1:
            cam = st.camera_input("Ambil Foto")
        with c2:
            if cam:
                img = Image.open(cam)
                lbl, conf = prediksi_gambar_diam(img)
                st.image(img, width=250)
                tampilkan_kartu_edukasi(lbl)
                if st.button("💾 Simpan Hasil"):
                    st.session_state['history_data'].append({"Waktu": datetime.now().strftime("%H:%M"), "Jenis": lbl})
                    st.toast("Tersimpan!")
            else:
                st.info("Silakan ambil foto sampah di sekitar kamu.")

    elif mode == "📹 Video Live":
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2:
            st.info("Arahkan sampah ke kotak di tengah layar 👇")
            webrtc_streamer(key="live", mode=WebRtcMode.SENDRECV,
                            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                            video_processor_factory=VideoProcessor,
                            media_stream_constraints={"video": True, "audio": False},
                            async_processing=True)

# === TAB 2: DASHBOARD ===
with tab_dash:
    st.subheader("📊 Statistik Sampah Terkumpul")
    if st.session_state['history_data']:
        df = pd.DataFrame(st.session_state['history_data'])
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Sampah", len(df))
        m2.metric("Organik", len(df[df['Jenis']=='ORGANIK']))
        m3.metric("Anorganik", len(df[df['Jenis']=='ANORGANIK']))
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig = px.pie(df, names='Jenis', hole=0.4, 
                         color='Jenis', color_discrete_map={'ORGANIK':'#4CAF50', 'ANORGANIK':'#FF9800'},
                         title="Persentase Jenis Sampah")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.write("##### 📝 Riwayat Deteksi")
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("Belum ada data. Yuk scan sampah dulu!")

# === TAB 3: INFO PROJECT ===
with tab_info:
    st.header("Tentang EcoSort Edu")
    st.write("Proyek ini bertujuan untuk membantu masyarakat memilah sampah dengan mudah menggunakan teknologi AI.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists("grafik_performa.png"):
            st.image("grafik_performa.png", caption="Grafik Akurasi Model AI", use_container_width=True)
    with col_b:
        st.info("Aplikasi ini menggunakan model Deep Learning CNN yang dilatih dengan dataset sampah organik & anorganik.")
