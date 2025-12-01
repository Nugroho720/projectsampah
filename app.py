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
import requests
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="EcoSort Premium",
    page_icon="♻️",
    layout="wide",
)

# --- 2. CSS PREMIUM (TAMPILAN CANTIK & BERSIH) ---
st.markdown("""
<style>
    /* Background Halaman */
    .stApp {
        background-color: #f4f9f4; /* Hijau Mint Sangat Muda */
    }
    
    /* Styling Container Putih (Kartu) */
    .css-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    
    /* Judul Utama */
    h1, h2, h3 {
        color: #2e7d32 !important;
        font-family: 'Helvetica', sans-serif;
        font-weight: 700;
    }
    
    /* Teks Biasa */
    p, div, label, span {
        color: #333333 !important;
    }

    /* Tombol Cantik */
    .stButton>button {
        background: linear-gradient(90deg, #4CAF50 0%, #2E7D32 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46, 125, 50, 0.3);
    }

    /* Highlight Hasil */
    .result-box-organik {
        background-color: #e8f5e9;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
    }
    .result-box-anorganik {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. DATABASE & LOGIKA AI ---

# Database Info Edukasi
info_sampah = {
    "ORGANIK": {
        "judul": "🌱 ORGANIK",
        "desc": "Sampah sisa makhluk hidup yang mudah terurai.",
        "saran": "Olah menjadi pupuk kompos atau pakan ternak."
    },
    "ANORGANIK": {
        "judul": "⚙️ ANORGANIK",
        "desc": "Sampah buatan pabrik yang sulit/tidak bisa terurai.",
        "saran": "Pisahkan untuk didaur ulang (Bank Sampah)."
    }
}

# Session State untuk History
if 'history_data' not in st.session_state:
    st.session_state['history_data'] = []

@st.cache_resource
def load_model():
    if not os.path.exists('model_sampah.h5'): return None
    return tf.keras.models.load_model('model_sampah.h5')

model = load_model()

# Class Video Processor (Dioptimalkan)
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if model is not None:
            # Resize kecil dulu biar cepat prosesnya
            img_small = cv2.resize(img, (64, 64))
            img_input = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            img_input = img_input.astype("float32") / 255.0
            img_input = np.expand_dims(img_input, axis=0)

            prediction = model.predict(img_input)
            confidence = prediction[0][0]

            if confidence > 0.5:
                label = "ORGANIK"
                color = (0, 200, 0) # Hijau
                prob = confidence
            else:
                label = "ANORGANIK"
                color = (0, 100, 255) # Oranye
                prob = 1 - confidence

            # Gambar GUI
            cv2.rectangle(img, (0, 0), (220, 50), (255, 255, 255), -1) 
            cv2.putText(img, f"{label}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Kotak Bidik
            h, w, _ = img.shape
            cv2.rectangle(img, (w//2-80, h//2-80), (w//2+80, h//2+80), color, 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def prediksi_gambar(image):
    img_resized = image.resize((64, 64))
    img_array = np.array(img_resized)
    if img_array.shape == (64, 64): img_array = np.stack((img_array,)*3, axis=-1)
    if img_array.shape[-1] == 4: img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0) / 255.0 
    prediction = model.predict(img_array)
    skor = prediction[0][0]
    return ("ORGANIK", skor) if skor > 0.5 else ("ANORGANIK", 1 - skor)

# ================= MAIN UI LAYOUT =================

# Header
c1, c2 = st.columns([1, 6])
with c1:
    st.image("https://cdn-icons-png.flaticon.com/512/3299/3299956.png", width=80)
with c2:
    st.title("EcoSort Premium")
    st.caption("Sistem Cerdas Klasifikasi Sampah & Manajemen Lingkungan")

st.write("---")

if model is None:
    st.error("⚠️ Model AI tidak ditemukan! Pastikan file 'model_sampah.h5' ada di GitHub.")
    st.stop()

# TABS UTAMA
tabs = st.tabs(["📸 SCANNER", "📊 DASHBOARD", "📚 ENSIKLOPEDIA"])

# --- TAB 1: SCANNER ---
with tabs[0]:
    col_kiri, col_kanan = st.columns([1, 1.5])
    
    with col_kiri:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("1. Input Gambar")
        pilihan = st.radio("Metode:", ["📂 Upload File", "📸 Ambil Foto", "📹 Live Video"], horizontal=True)
        
        image_input = None
        
        if pilihan == "📂 Upload File":
            uploaded = st.file_uploader("Pilih file...", type=['jpg','png','jpeg'])
            if uploaded: image_input = Image.open(uploaded)
            
        elif pilihan == "📸 Ambil Foto":
            cam = st.camera_input("Jepret sekarang")
            if cam: image_input = Image.open(cam)
            
        elif pilihan == "📹 Live Video":
            st.info("💡 Koneksi Live Video tergantung kecepatan internet.")
            webrtc_streamer(key="live", mode=WebRtcMode.SENDRECV,
                            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                            video_processor_factory=VideoProcessor,
                            media_stream_constraints={"video": True, "audio": False},
                            async_processing=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_kanan:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("2. Hasil Analisis AI")
        
        if image_input:
            # Tampilkan Gambar
            st.image(image_input, caption="Preview Input", width=300)
            st.write("---")
            
            # Proses AI
            label, conf = prediksi_gambar(image_input)
            info = info_sampah[label]
            
            # Kotak Hasil Warna-warni
            css_class = "result-box-organik" if label == "ORGANIK" else "result-box-anorganik"
            st.markdown(f"""
            <div class="{css_class}">
                <h2 style="margin:0;">{info['judul']}</h2>
                <p><b>Akurasi:</b> {conf*100:.1f}%</p>
                <p>{info['desc']}</p>
                <p>💡 <b>Saran:</b> {info['saran']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            if st.button("💾 Simpan Data ke Dashboard"):
                st.session_state['history_data'].append({
                    "Waktu": datetime.now().strftime("%H:%M:%S"),
                    "Jenis": label,
                    "Akurasi": f"{conf*100:.1f}%"
                })
                st.toast("Data berhasil disimpan!", icon="✅")
        else:
            if pilihan != "📹 Live Video":
                st.info("👈 Silakan masukkan gambar di menu sebelah kiri.")
            else:
                st.write("Sedang menjalankan mode video...")
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: DASHBOARD ---
with tabs[1]:
    st.markdown("### 📊 Statistik Sampah Terkumpul")
    
    if len(st.session_state['history_data']) > 0:
        df = pd.DataFrame(st.session_state['history_data'])
        
        # Metrik Atas
        m1, m2, m3 = st.columns(3)
        with m1: 
            st.markdown('<div class="css-card"><h3>Total</h3><h1>'+str(len(df))+'</h1></div>', unsafe_allow_html=True)
        with m2:
            org = len(df[df['Jenis']=='ORGANIK'])
            st.markdown('<div class="css-card"><h3 style="color:green!important">Organik</h3><h1>'+str(org)+'</h1></div>', unsafe_allow_html=True)
        with m3:
            anorg = len(df[df['Jenis']=='ANORGANIK'])
            st.markdown('<div class="css-card"><h3 style="color:orange!important">Anorganik</h3><h1>'+str(anorg)+'</h1></div>', unsafe_allow_html=True)

        st.write("")
        
        # Grafik & Tabel
        c_chart, c_table = st.columns([1, 1.5])
        with c_chart:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            fig = px.pie(df, names='Jenis', title='Persentase Sampah',
                         color='Jenis', color_discrete_map={'ORGANIK':'#4CAF50', 'ANORGANIK':'#FF9800'}, hole=0.5)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c_table:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.subheader("📝 Riwayat Deteksi Terbaru")
            st.dataframe(df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Data kosong. Silakan scan sampah dulu di menu Scanner!")

# --- TAB 3: ENSIKLOPEDIA ---
with tabs[2]:
    st.markdown("### 📚 Ensiklopedia Sampah")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<div class="css-card" style="border-top: 5px solid green;">', unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/2909/2909796.png", width=100)
        st.subheader("Tentang Organik")
        st.write("""
        Sampah organik adalah sampah yang berasal dari sisa-sisa makhluk hidup, 
        baik hewan, tanaman, maupun manusia.
        
        **Ciri-ciri:**
        - Mudah membusuk
        - Mengandung air
        - Dapat diurai oleh mikroorganisme
        
        **Contoh:** Kulit pisang, duri ikan, sayuran layu.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="css-card" style="border-top: 5px solid orange;">', unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/9632/9632631.png", width=100)
        st.subheader("Tentang Anorganik")
        st.write("""
        Sampah anorganik adalah sampah yang dihasilkan dari bahan-bahan non-hayati,
        baik berupa produk sintetik maupun hasil proses teknologi.
        
        **Ciri-ciri:**
        - Sangat sulit terurai (bisa ratusan tahun)
        - Tahan air dan cuaca
        - Bisa didaur ulang (Recycle)
        
        **Contoh:** Botol plastik, kaleng soda, kaca, styrofoam.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
