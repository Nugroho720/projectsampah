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
    page_title="EcoSort Edu",
    page_icon="♻️",
    layout="wide",
)

# --- 2. CSS "PLATINUM STYLE" (TOSCA & GLASSMORPHISM) ---
st.markdown("""
<style>
    /* 1. BACKGROUND GRADASI TOSCA (#40E0D0) */
    .stApp {
        background: linear-gradient(180deg, #40E0D0 0%, #E0F7FA 100%);
        background-attachment: fixed;
    }
    
    /* 2. PAKSA SEMUA TEKS JADI HITAM (ANTI DARK MODE) */
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, div, span, label, li {
        color: #1a1a1a !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* 3. CONTAINER PUTIH KACA (GLASSMORPHISM) */
    .css-card {
        background: rgba(255, 255, 255, 0.85); /* Putih Transparan */
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 20px;
    }
    
    /* 4. SIDEBAR PUTIH BERSIH */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #ddd;
    }
    
    /* 5. TOMBOL TOSCA GRADASI */
    .stButton>button {
        background: linear-gradient(90deg, #008080 0%, #40E0D0 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(64, 224, 208, 0.4);
    }

    /* KOTAK INFO EDUKASI */
    .info-box-organik {
        background-color: #E8F5E9;
        border-left: 6px solid #2E7D32;
        padding: 15px;
        border-radius: 10px;
    }
    .info-box-anorganik {
        background-color: #FFF3E0;
        border-left: 6px solid #EF6C00;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGIKA BACKEND ---
info_sampah = {
    "ORGANIK": {
        "judul": "🌱 ORGANIK",
        "sifat": "Mudah Terurai (Biodegradable)",
        "desc": "Berasal dari alam (sisa makhluk hidup). Bisa jadi kompos.",
        "aksi": "Olah jadi **Pupuk Kompos** atau **Pakan Ternak**."
    },
    "ANORGANIK": {
        "judul": "⚙️ ANORGANIK",
        "sifat": "Sulit Terurai (Non-Biodegradable)",
        "desc": "Bahan sintesis/pabrikan. Tahan ratusan tahun.",
        "aksi": "Wajib **Daur Ulang (Recycle)** di Bank Sampah."
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
lottie_main = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_tutvdkg0.json")

if 'history_data' not in st.session_state: st.session_state['history_data'] = []

# --- 4. VIDEO PROCESSOR CANGGIH (DENGAN MIRRORING) ---
class VideoProcessor:
    def __init__(self):
        self.mirror = False # Default status

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Fitur Mirroring (Membalik Gambar)
        if self.mirror:
            img = cv2.flip(img, 1)

        if model is not None:
            # Preprocessing Cepat
            img_small = cv2.resize(img, (64, 64))
            img_input = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            img_input = img_input.astype("float32") / 255.0
            img_input = np.expand_dims(img_input, axis=0)

            prediction = model.predict(img_input)
            confidence = prediction[0][0]

            if confidence > 0.5:
                label, color = "ORGANIK", (0, 255, 0) # Hijau
                prob = confidence
            else:
                label, color = "ANORGANIK", (0, 140, 255) # Oranye
                prob = 1 - confidence

            # HUD (Tampilan Layar Canggih)
            # Kotak Label Semi-Transparan
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (250, 70), (0, 0, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            cv2.putText(img, f"{label}", (10, 35), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            cv2.putText(img, f"Akurasi: {prob*100:.0f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Kotak Bidik Tengah (Crosshair)
            h, w, _ = img.shape
            cv2.line(img, (w//2-20, h//2), (w//2+20, h//2), color, 2)
            cv2.line(img, (w//2, h//2-20), (w//2, h//2+20), color, 2)
            cv2.rectangle(img, (w//2-100, h//2-100), (w//2+100, h//2+100), color, 2)

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

# ================= MAIN APP =================

# SIDEBAR (BRANDING)
with st.sidebar:
    if lottie_main: st_lottie(lottie_main, height=150, key="anim")
    st.markdown("## EcoSort Edu 🌿")
    st.markdown("*Aplikasi Klasifikasi Sampah Cerdas*")
    st.divider()
    st.info("💡 **Tips:** Pastikan objek sampah terlihat jelas dan terang.")
    st.caption("© 2025 Project UAS")

if model is None:
    st.error("⚠️ Model belum ditemukan di GitHub.")
    st.stop()

# JUDUL UTAMA (HEADER)
c1, c2 = st.columns([0.5, 4])
with c1:
    st.image("https://cdn-icons-png.flaticon.com/512/3299/3299956.png", width=70)
with c2:
    st.title("EcoSort Edu")
    st.markdown("#### Sistem Klasifikasi & Edukasi Sampah Berbasis AI")

# TABS
tab1, tab2, tab3 = st.tabs(["📸 SCANNER", "📊 DASHBOARD", "ℹ️ TENTANG"])

# === TAB 1: SCANNER ===
with tab1:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.write("##### 1. Pilih Metode Input:")
    mode = st.radio("", ["📂 Upload File", "📸 Kamera Foto", "📹 Live Video (Canggih)"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_kiri, col_kanan = st.columns([1.5, 1])

    with col_kiri:
        if mode == "📂 Upload File":
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            up = st.file_uploader("Tarik gambar kesini...", type=['jpg','png','jpeg'])
            if up:
                img = Image.open(up)
                st.image(img, use_container_width=True)
                lbl, conf = prediksi_gambar_diam(img)
                # Simpan variable untuk ditampilkan di kanan
                st.session_state['last_result'] = (lbl, conf)
            st.markdown('</div>', unsafe_allow_html=True)

        elif mode == "📸 Kamera Foto":
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            cam = st.camera_input("Jepret Sampah")
            if cam:
                img = Image.open(cam)
                lbl, conf = prediksi_gambar_diam(img)
                st.session_state['last_result'] = (lbl, conf)
            st.markdown('</div>', unsafe_allow_html=True)

        elif mode == "📹 Live Video (Canggih)":
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            
            # FITUR MIRRORING KAMERA
            is_mirror = st.checkbox("🔄 Balik Kamera (Mirroring)", value=False)
            
            ctx = webrtc_streamer(
                key="live", 
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
            # Mengirim data Mirror ke Processor
            if ctx.video_processor:
                ctx.video_processor.mirror = is_mirror
                
            st.caption("Jika video lambat, gunakan mode 'Kamera Foto' untuk hasil terbaik.")
            st.markdown('</div>', unsafe_allow_html=True)

    # PANEL HASIL (KANAN)
    with col_kanan:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.write("##### 2. Hasil Analisis AI")
        
        # Cek apakah ada hasil dari Upload/Foto
        if mode != "📹 Live Video (Canggih)":
            if 'last_result' in st.session_state and (mode == "📂 Upload File" and up) or (mode == "📸 Kamera Foto" and cam):
                lbl, conf = st.session_state['last_result']
                info = info_sampah[lbl]
                
                # Desain Kotak Hasil
                css_box = "info-box-organik" if lbl == "ORGANIK" else "info-box-anorganik"
                
                st.markdown(f"""
                <div class="{css_box}">
                    <h2 style="margin:0; color:#1a1a1a;">{info['judul']}</h2>
                    <p style="font-size:1.1em;"><b>{info['sifat']}</b></p>
                    <p>{info['desc']}</p>
                    <hr>
                    <p>💡 <b>Saran:</b> {info['aksi']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Tingkat Keyakinan", f"{conf*100:.1f}%")
                
                if st.button("💾 Simpan Data"):
                    st.session_state['history_data'].append({"Waktu": datetime.now().strftime("%H:%M"), "Jenis": lbl})
                    st.toast("Data berhasil disimpan!", icon="✅")
            else:
                st.info("👈 Masukkan gambar di sebelah kiri.")
        else:
            st.info("Lihat hasil deteksi langsung pada layar video di samping 👈")
            
        st.markdown('</div>', unsafe_allow_html=True)

# === TAB 2: DASHBOARD ===
with tab2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    if st.session_state['history_data']:
        df = pd.DataFrame(st.session_state['history_data'])
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(df))
        c2.metric("Organik", len(df[df['Jenis']=='ORGANIK']))
        c3.metric("Anorganik", len(df[df['Jenis']=='ANORGANIK']))
        
        col_g, col_t = st.columns([1, 1])
        with col_g:
            fig = px.pie(df, names='Jenis', hole=0.5, color='Jenis', 
                         color_discrete_map={'ORGANIK':'#2E7D32', 'ANORGANIK':'#EF6C00'})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with col_t:
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("Data kosong. Yuk scan sampah dulu!")
    st.markdown('</div>', unsafe_allow_html=True)

# === TAB 3: TENTANG ===
with tab3:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("Tentang EcoSort Edu")
    st.write("Aplikasi ini dibuat untuk memenuhi Tugas UAS Pengenalan Pola.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists("grafik_performa.png"):
            st.image("grafik_performa.png", caption="Training Accuracy Graph", use_container_width=True)
    with col_b:
        st.success("**Teknologi:** Python, TensorFlow CNN, Streamlit, OpenCV.")
        st.info("**Developer:** [Nama Kamu]")
    st.markdown('</div>', unsafe_allow_html=True)
