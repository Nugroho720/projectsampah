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

# --- 2. CSS PLATINUM (TOSCA GRADIENT + BERSIH) ---
st.markdown("""
<style>
    /* BACKGROUND GRADASI TOSCA */
    .stApp {
        background: linear-gradient(180deg, #A7FFEB 0%, #E0F7FA 100%);
        background-attachment: fixed;
    }
    
    /* MENYEMBUNYIKAN HEADER BAWAAN STREAMLIT BIAR BERSIH */
    header {visibility: hidden;}
    
    /* MEMAKSA TEKS JADI HITAM JELAS (ANTI DARK MODE) */
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, div, span, label, li {
        color: #004D40 !important; /* Hijau Tua Gelap */
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #B2DFDB;
    }
    
    /* TOMBOL UPDATE */
    .stButton>button {
        background: linear-gradient(45deg, #009688, #4DB6AC);
        color: white !important;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }

    /* KARTU INFO HASIL */
    .success-box {
        padding: 20px;
        background-color: #E0F2F1;
        border-left: 6px solid #009688;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .warning-box {
        padding: 20px;
        background-color: #FFF3E0;
        border-left: 6px solid #FF9800;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    
    /* MENGHILANGKAN KOTAK PUTIH KOSONG YANG MENGGANGGU */
    .css-card { display: none; }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA & MODEL ---
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

# Bank Soal Kuis (Lebih Banyak & Edukatif)
kuis_db = [
    {
        "tanya": "1. Kulit pisang yang dibuang ke tanah akan terurai dalam waktu...",
        "opsi": ["A. 100 Tahun", "B. 2-5 Minggu", "C. Tidak bisa terurai"],
        "kunci": "B. 2-5 Minggu",
        "info": "Benar! Sampah organik seperti kulit buah sangat cepat terurai oleh bakteri tanah."
    },
    {
        "tanya": "2. Manakah di bawah ini yang termasuk sampah B3 (Bahan Berbahaya Beracun)?",
        "opsi": ["A. Baterai Bekas", "B. Botol Plastik", "C. Kertas Koran"],
        "kunci": "A. Baterai Bekas",
        "info": "Tepat! Baterai mengandung zat kimia berbahaya dan tidak boleh dibuang sembarangan."
    },
    {
        "tanya": "3. Botol plastik air mineral sebaiknya diperlakukan bagaimana?",
        "opsi": ["A. Dibakar", "B. Dibuang ke sungai", "C. Didaur Ulang (Recycle)"],
        "kunci": "C. Didaur Ulang (Recycle)",
        "info": "Betul. Membakar plastik berbahaya bagi pernapasan, mendaur ulang adalah solusi terbaik."
    },
    {
        "tanya": "4. Apa warna tempat sampah yang umum digunakan untuk sampah Organik?",
        "opsi": ["A. Hijau", "B. Kuning", "C. Merah"],
        "kunci": "A. Hijau",
        "info": "Benar. Hijau untuk Organik, Kuning untuk Anorganik, Merah untuk B3."
    },
    {
        "tanya": "5. Styrofoam membutuhkan waktu berapa lama untuk hancur alami?",
        "opsi": ["A. 1 Tahun", "B. 50 Tahun", "C. Tidak dapat terurai / >500 tahun"],
        "kunci": "C. Tidak dapat terurai / >500 tahun",
        "info": "Benar sekali. Styrofoam adalah musuh lingkungan karena hampir abadi!"
    }
]

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
lottie_quiz = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_zrqthn6o.json")
lottie_main = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_tutvdkg0.json")

if 'history_data' not in st.session_state: st.session_state['history_data'] = []

# --- 4. VIDEO PROCESSOR (MIRRORING FIX) ---
class VideoProcessor:
    def __init__(self):
        self.mirror = False 

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.mirror:
            img = cv2.flip(img, 1)

        if model is not None:
            img_small = cv2.resize(img, (64, 64))
            img_input = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            img_input = img_input.astype("float32") / 255.0
            img_input = np.expand_dims(img_input, axis=0)

            prediction = model.predict(img_input)
            confidence = prediction[0][0]

            if confidence > 0.5:
                label, color = "ORGANIK", (0, 255, 0)
                prob = confidence
            else:
                label, color = "ANORGANIK", (0, 140, 255)
                prob = 1 - confidence

            cv2.rectangle(img, (0, 0), (280, 60), (255, 255, 255), -1)
            cv2.putText(img, f"{label} ({prob*100:.0f}%)", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
            
            h, w, _ = img.shape
            cv2.rectangle(img, (w//2-80, h//2-80), (w//2+80, h//2+80), color, 3)

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

# ================= MAIN UI =================

# SIDEBAR
with st.sidebar:
    if lottie_main: st_lottie(lottie_main, height=150, key="anim")
    st.markdown("## EcoSort Edu 🌿")
    st.markdown("**Versi Pelajar Pro**")
    st.divider()
    st.info("💡 **Info:** Aplikasi ini membantu mengidentifikasi jenis sampah untuk pemilahan yang lebih baik.")
    st.caption("© 2025 Project UAS")

if model is None:
    st.error("⚠️ Model belum ditemukan di GitHub.")
    st.stop()

# HEADER
c1, c2 = st.columns([0.5, 4])
with c1:
    st.image("https://cdn-icons-png.flaticon.com/512/3299/3299956.png", width=80)
with c2:
    st.title("EcoSort Edu")
    st.markdown("### Klasifikasi & Edukasi Sampah")

# TABS NAVIGATION
tab1, tab2, tab3, tab4 = st.tabs(["📸 SCANNER", "📊 DATA STATISTIK", "🎓 KUIS EDUKASI", "ℹ️ INFO PROJECT"])

# === TAB 1: SCANNER ===
with tab1:
    st.markdown("#### 1. Pilih Metode Input:")
    mode = st.radio("", ["📁 Upload File", "📸 Kamera Foto", "📹 Live Video"], horizontal=True)
    st.write("---")

    col_kiri, col_kanan = st.columns([1.5, 1])

    with col_kiri:
        if mode == "📁 Upload File":
            up = st.file_uploader("Upload gambar sampah...", type=['jpg','png','jpeg'])
            if up:
                img = Image.open(up)
                st.image(img, caption="Preview Gambar", use_container_width=True)
                # Simpan di session state
                st.session_state['last_img'] = img
                st.session_state['last_mode'] = 'upload'

        elif mode == "📸 Kamera Foto":
            cam = st.camera_input("Jepret Sampah")
            if cam:
                img = Image.open(cam)
                st.session_state['last_img'] = img
                st.session_state['last_mode'] = 'camera'

        elif mode == "📹 Live Video":
            # FITUR MIRRORING
            mirror_mode = st.checkbox("🔄 Balik Kamera (Mirror)", value=True)
            
            ctx = webrtc_streamer(
                key="live", 
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
            if ctx.video_processor:
                ctx.video_processor.mirror = mirror_mode

    # PANEL HASIL (KANAN)
    with col_kanan:
        st.markdown("#### 2. Hasil Analisis")
        
        # Logika Menampilkan Hasil
        if mode != "📹 Live Video":
            if 'last_img' in st.session_state and st.session_state.get('last_mode') in ['upload', 'camera']:
                # Proses Gambar
                img = st.session_state['last_img']
                lbl, conf = prediksi_gambar_diam(img)
                
                # Tampilan Kotak Keren
                info = info_sampah[lbl]
                css_box = "success-box" if lbl == "ORGANIK" else "warning-box"
                
                st.markdown(f"""
                <div class="{css_box}">
                    <h2 style="margin-top:0; color:#004D40;">{info['judul']}</h2>
                    <p style="font-size:1.1em;"><b>{info['sifat']}</b></p>
                    <p>{info['desc']}</p>
                    <hr>
                    <p>💡 <b>Saran:</b> {info['aksi']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Tingkat Keyakinan AI", f"{conf*100:.1f}%")
                
                if st.button("💾 Simpan Data"):
                    st.session_state['history_data'].append({"Waktu": datetime.now().strftime("%H:%M"), "Jenis": lbl})
                    st.toast("Data berhasil disimpan!", icon="✅")
            else:
                st.info("👈 Masukkan gambar di sebelah kiri untuk melihat hasil.")
        else:
            st.info("Lihat hasil deteksi langsung pada layar video 👈")

# === TAB 2: DASHBOARD ===
with tab2:
    if st.session_state['history_data']:
        df = pd.DataFrame(st.session_state['history_data'])
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Sampah", len(df))
        c2.metric("Organik", len(df[df['Jenis']=='ORGANIK']))
        c3.metric("Anorganik", len(df[df['Jenis']=='ANORGANIK']))
        
        col_chart, col_data = st.columns([1, 1])
        with col_chart:
            fig = px.pie(df, names='Jenis', hole=0.4, color='Jenis', 
                         color_discrete_map={'ORGANIK':'#00C853', 'ANORGANIK':'#FF6D00'})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with col_data:
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("Belum ada data tersimpan. Yuk scan sampah dulu!")

# === TAB 3: KUIS EDUKASI (FITUR BARU & LEBIH BANYAK) ===
with tab3:
    col_q, col_anim = st.columns([2, 1])
    with col_q:
        st.markdown("### 🧠 Tantangan Pengetahuan Lingkungan")
        st.write("Jawab pertanyaan berikut untuk menguji wawasanmu!")
    with col_anim:
        if lottie_quiz: st_lottie(lottie_quiz, height=150)
    
    st.divider()
    
    score = 0
    # Loop pertanyaan
    for i, q in enumerate(kuis_db):
        st.markdown(f"**{q['tanya']}**")
        jawaban = st.radio(f"Pilih jawaban nomor {i+1}:", q['opsi'], key=f"q{i}", index=None)
        
        if jawaban:
            if jawaban == q['kunci']:
                st.success(f"✅ Benar! {q['info']}")
                score += 20 # 5 Soal x 20 = 100
            else:
                st.error("❌ Kurang tepat.")
        st.write("---")
    
    # Hasil Akhir
    if st.button("Hitung Skor Saya"):
        if score == 100:
            st.balloons()
            st.markdown(f"### 🎉 LUAR BIASA! Skor Kamu: {score}/100")
            st.info("Kamu adalah pahlawan lingkungan sejati!")
        elif score >= 60:
            st.markdown(f"### 👏 BAGUS! Skor Kamu: {score}/100")
            st.write("Tingkatkan lagi wawasanmu ya.")
        else:
            st.markdown(f"### Skor Kamu: {score}/100")
            st.write("Jangan menyerah, ayo belajar lagi di menu Scanner!")

# === TAB 4: INFO PROJECT ===
with tab4:
    st.header("Tentang Aplikasi")
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("grafik_performa.png"):
            st.image("grafik_performa.png", caption="Grafik Akurasi Model AI", use_container_width=True)
    with c2:
        st.write("""
        **EcoSort Edu** adalah aplikasi berbasis Artificial Intelligence untuk membantu
        edukasi pemilahan sampah di masyarakat.
        
        **Fitur Unggulan:**
        - ✨ Deteksi Real-time dengan CNN
        - 📱 Support Kamera HP & Laptop
        - 🎓 Modul Edukasi Interaktif
        """)
        st.caption("Dibuat dengan ❤️ menggunakan Python & Streamlit")
