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
    page_title="EcoSort Edu - Smart Waste App",
    page_icon="🌱",
    layout="wide",
)

# --- 2. DESAIN UI "ECO-FRIENDLY" (CSS INJECTION) ---
st.markdown("""
<style>
    /* Mengubah Background Utama jadi Mint Cream lembut */
    .stApp {
        background-color: #F1F8E9;
    }
    
    /* Sidebar Putih Bersih */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 2px solid #C8E6C9;
    }

    /* Judul Halaman (H1, H2, H3) jadi Hijau Hutan */
    h1, h2, h3 {
        color: #2E7D32 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Mempercantik Tombol (Hijau Cerah) */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Kotak Metrik/Info Dashboard */
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
    }

    /* Info Box Edukasi */
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
lottie_quiz = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_zrqthn6o.json") # Animasi baru buat kuis

if 'history_data' not in st.session_state: st.session_state['history_data'] = []
if 'quiz_score' not in st.session_state: st.session_state['quiz_score'] = 0

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
                label, color = "ANORGANIK", (0, 100, 255) # Oranye BGR
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
        <h3 style="color: #2E7D32; margin-top:0;">{data['judul']}</h3>
        <p style="font-size: 1.1em;"><b>{data['sifat']}</b></p>
        <p>{data['definisi']}</p>
        <hr>
        <p>💡 <b>Saran Pengelolaan:</b> {data['aksi']}</p>
    </div>
    """, unsafe_allow_html=True)

# ================= MAIN APP =================

# SIDEBAR
with st.sidebar:
    if lottie_recycle: st_lottie(lottie_recycle, height=180, key="anim_sidebar")
    st.title("EcoSort Edu 🌿")
    st.markdown("Aplikasi pintar untuk masa depan bumi yang lebih hijau.")
    st.divider()
    st.info("💡 **Tips Hari Ini:**\nJangan lupa memisahkan tutup botol plastik sebelum membuangnya!")
    st.caption("© 2025 Project UAS [Nama Kamu]")

if model is None:
    st.error("🚨 Model tidak ditemukan. Harap jalankan training dulu!")
    st.stop()

# JUDUL UTAMA (Dihias icon)
st.title("🌍 Klasifikasi & Edukasi Sampah")
st.markdown("**Selamat datang!** Mari belajar memilah sampah dengan kecerdasan buatan.")

# TABS
tab_scan, tab_dash, tab_quiz, tab_info = st.tabs(["📸 AI Scanner", "📊 Statistik", "🎮 Kuis Sampah", "ℹ️ Info Project"])

# === TAB 1: SCANNER ===
with tab_scan:
    st.markdown("#### 🔍 Pilih Metode Deteksi:")
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
    if st.session_state['history_data']:
        df = pd.DataFrame(st.session_state['history_data'])
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Sampah", len(df))
        m2.metric("Organik", len(df[df['Jenis']=='ORGANIK']))
        m3.metric("Anorganik", len(df[df['Jenis']=='ANORGANIK']))
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            # Grafik Pie Chart dengan warna Custom (Hijau & Oranye)
            fig = px.pie(df, names='Jenis', hole=0.4, 
                         color='Jenis', color_discrete_map={'ORGANIK':'#4CAF50', 'ANORGANIK':'#FF9800'},
                         title="Persentase Jenis Sampah")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.write("##### 📝 Riwayat Deteksi")
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("Belum ada data. Yuk scan sampah dulu!")

# === TAB 3: KUIS SAMPAH (FITUR BARU) ===
with tab_quiz:
    c1, c2 = st.columns([1, 2])
    with c1:
        if lottie_quiz: st_lottie(lottie_quiz, height=200)
    with c2:
        st.subheader("🧠 Uji Pengetahuanmu!")
        st.write("Jawab pertanyaan berikut untuk melihat seberapa paham kamu tentang pemilahan sampah.")

    st.divider()
    
    # Soal 1
    q1 = st.radio("1. Kulit pisang termasuk jenis sampah apa?", ["Anorganik", "Organik", "B3"], index=None)
    if q1:
        if q1 == "Organik": st.success("Benar! Kulit pisang mudah membusuk.")
        else: st.error("Salah. Kulit pisang berasal dari alam, jadi itu Organik.")

    # Soal 2
    st.write("---")
    q2 = st.radio("2. Berapa lama botol plastik bisa terurai?", ["1 minggu", "1 tahun", "450 tahun"], index=None)
    if q2:
        if q2 == "450 tahun": st.success("Tepat! Plastik sangat berbahaya karena lama terurai.")
        else: st.error("Kurang tepat. Plastik butuh waktu sangat lama (ratusan tahun).")

    # Soal 3
    st.write("---")
    q3 = st.radio("3. Manakah yang BUKAN sampah anorganik?", ["Kaleng", "Daun Kering", "Kaca"], index=None)
    if q3:
        if q3 == "Daun Kering": st.success("Betul! Daun kering adalah sampah organik.")
        else: st.error("Salah, coba lagi ya!")

# === TAB 4: INFO PROJECT ===
with tab_info:
    st.header("Tentang EcoSort Edu")
    st.write("Proyek ini bertujuan untuk membantu masyarakat memilah sampah dengan mudah menggunakan teknologi AI.")
    if os.path.exists("grafik_performa.png"):
        st.image("grafik_performa.png", caption="Grafik Akurasi Model AI", width=600)
