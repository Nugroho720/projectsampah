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
    page_icon="🌍",
    layout="wide",
)

# --- 2. CSS & TEMA "CLEAN TOSCA" ---
st.markdown("""
<style>
    /* BACKGROUND GRADASI HIJAU TOSCA */
    .stApp {
        background: linear-gradient(180deg, #A7FFEB 0%, #E0F7FA 100%);
        background-attachment: fixed;
    }
    
    /* HIDE HEADER BAWAAN */
    header {visibility: hidden;}
    
    /* TEKS HITAM JELAS (ANTI DARK MODE) */
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, div, span, label, li {
        color: #004D40 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #B2DFDB;
    }
    
    /* TOMBOL UPDATE */
    .stButton>button {
        background: linear-gradient(45deg, #00BFA5, #1DE9B6);
        color: white !important;
        border-radius: 12px;
        border: none;
        font-weight: bold;
        transition: 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }

    /* KOTAK INFO EDUKASI */
    .info-card {
        padding: 25px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.95);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 8px solid #00BFA5;
    }
    
    /* CUSTOM HEADER (TRANSPARAN / TANPA KOTAK PUTIH) */
    .custom-header {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        margin-bottom: 20px;
        flex-wrap: wrap;
        /* Background dihapus biar menyatu */
    }
    .header-text {
        margin-left: 20px;
        text-align: left;
    }
    
    /* RESPONSIF HP */
    @media (max-width: 600px) {
        .custom-header { flex-direction: column; text-align: center; }
        .header-text { margin-left: 0; margin-top: 10px; text-align: center; }
    }
    
    /* MENGHILANGKAN KOTAK KOSONG */
    .css-card { display: none; }
</style>
""", unsafe_allow_html=True)

# --- 3. DATABASE EDUKASI ---
info_sampah = {
    "ORGANIK": {
        "judul": "🌱 SAMPAH ORGANIK",
        "sifat": "✅ Mudah Terurai (Biodegradable)",
        "desc": "Sampah ini berasal dari alam. Jika dibiarkan di tanah, ia akan membusuk dan menyatu dengan bumi.",
        "bahaya": "⚠️ **Hati-hati:** Jika sampah ini ditumpuk di dalam plastik tertutup (TPA), ia akan menghasilkan **Gas Metana** yang menyebabkan Pemanasan Global (Global Warming)!",
        "manfaat": "✨ **Solusi Emas:** Jangan buang percuma! Olah menjadi **Pupuk Kompos** untuk menyuburkan tanaman ibumu, atau jadikan **Eco-Enzyme** pembersih alami.",
        "aksi": "Masukkan ke lubang biopori atau komposter rumah."
    },
    "ANORGANIK": {
        "judul": "⚙️ SAMPAH ANORGANIK",
        "sifat": "❌ Sulit Terurai (Bisa Ratusan Tahun)",
        "desc": "Sampah buatan manusia/pabrik. Bakteri pengurai tidak doyan sampah ini.",
        "bahaya": "⚠️ **Bahaya Fatal:** Jika dibuang ke sungai, ia menyumbat dan bikin banjir. Jika dibuang ke laut, ia dimakan ikan dan menjadi **Mikroplastik** yang akhirnya kita makan juga!",
        "manfaat": "✨ **Solusi Cerdas:** Sampah ini adalah **Uang**! Kumpulkan, bersihkan, dan jual ke **Bank Sampah**. Atau kreasikan menjadi kerajinan tangan yang cantik.",
        "aksi": "Pisahkan dari sampah basah. Setorkan ke pemulung atau Bank Sampah."
    }
}

# --- DATABASE KUIS PER LEVEL ---
kuis_data = {
    "Level 1: Pemula (Basic)": [
        {"t": "Kulit jeruk termasuk jenis sampah apa?", "o": ["Organik", "Anorganik", "Residu"], "k": "Organik", "msg": "Betul! Kulit buah berasal dari alam."},
        {"t": "Warna tempat sampah untuk organik biasanya...", "o": ["Merah", "Hijau", "Biru"], "k": "Hijau", "msg": "Tepat! Hijau melambangkan daun/alam."},
        {"t": "Botol plastik bekas minum sebaiknya...", "o": ["Dibuang ke sungai", "Dibakar", "Didaur Ulang"], "k": "Didaur Ulang", "msg": "Benar. Daur ulang menyelamatkan bumi!"}
    ],
    "Level 2: Pengetahuan (Intermediate)": [
        {"t": "Berapa lama styrofoam hancur secara alami?", "o": ["10 Tahun", "100 Tahun", "Tidak bisa terurai"], "k": "Tidak bisa terurai", "msg": "Ya! Styrofoam adalah musuh abadi lingkungan."},
        {"t": "Gas berbahaya yang dihasilkan sampah organik di TPA adalah...", "o": ["Oksigen", "Metana", "Nitrogen"], "k": "Metana", "msg": "Tepat. Metana memperparah efek rumah kaca."},
        {"t": "Apa itu 3R dalam pengelolaan sampah?", "o": ["Run, Rush, Race", "Reduce, Reuse, Recycle", "Read, Rest, Relax"], "k": "Reduce, Reuse, Recycle", "msg": "Benar! Kurangi, Gunakan Ulang, Daur Ulang."}
    ],
    "Level 3: Ahli Lingkungan (Expert)": [
        {"t": "Sampah B3 (Bahan Berbahaya Beracun) contohnya adalah...", "o": ["Daun Kering", "Baterai Bekas", "Kertas"], "k": "Baterai Bekas", "msg": "Akurat. Baterai mengandung logam berat berbahaya."},
        {"t": "Apa itu 'Eco-Brick'?", "o": ["Bata dari tanah liat", "Botol plastik berisi padatan sampah plastik", "Tanaman hias"], "k": "Botol plastik berisi padatan sampah plastik", "msg": "Keren! Eco-brick solusi mengurangi limbah plastik."},
        {"t": "Mikroplastik adalah...", "o": ["Plastik ukuran besar", "Plastik baru", "Partikel plastik sangat kecil (<5mm)"], "k": "Partikel plastik sangat kecil (<5mm)", "msg": "Benar. Ini sangat berbahaya bagi kesehatan laut dan manusia."}
    ]
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
lottie_flower = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_u4yrau.json") 
lottie_sidebar = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_tutvdkg0.json")

if 'history_data' not in st.session_state: st.session_state['history_data'] = []

# --- 4. VIDEO PROCESSOR ---
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

# SIDEBAR (TETAP ADA)
with st.sidebar:
    if lottie_sidebar: st_lottie(lottie_sidebar, height=150, key="anim")
    st.markdown("## EcoSort Edu 🌿")
    st.markdown("*'Satu langkah kecil memilah, lompatan besar untuk bumi.'*")
    st.divider()
    st.info("💡 **Tips:** Pisahkan minyak jelantah, jangan buang di wastafel!")
    st.caption("© 2025 Project UAS")

if model is None:
    st.error("⚠️ Model belum ditemukan di GitHub.")
    st.stop()

# --- HEADER CUSTOM HTML (JUDUL MENYATU DENGAN BACKGROUND) ---
st.markdown("""
<div class="custom-header">
    <img src="https://cdn-icons-png.flaticon.com/512/2947/2947656.png" width="90">
    <div class="header-text">
        <h1 style="margin:0; font-size: 3rem; color:#004D40; font-weight:800;">EcoSort Edu</h1>
        <p style="margin:0; font-size: 1.3rem; color:#00695C; font-weight:500;">
        "Klasifikasi & Edukasi Sampah"
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# TABS NAVIGATION
tab1, tab2, tab3, tab4 = st.tabs(["📸 AI SCANNER", "📊 STATISTIK", "🎓 KUIS BERLEVEL", "ℹ️ INFO PROJECT"])

# === TAB 1: SCANNER ===
with tab1:
    st.markdown("#### 1. Pilih Metode Input:")
    mode = st.radio("", ["📁 Upload File", "📸 Kamera Foto", "📹 Live Video"], horizontal=True, label_visibility="collapsed")
    st.write("---")

    col_kiri, col_kanan = st.columns([1.5, 1])

    with col_kiri:
        if mode == "📁 Upload File":
            up = st.file_uploader("Upload gambar sampah...", type=['jpg','png','jpeg'])
            if up:
                img = Image.open(up)
                st.image(img, caption="Preview Gambar", use_container_width=True)
                st.session_state['last_img'] = img
                st.session_state['last_mode'] = 'upload'

        elif mode == "📸 Kamera Foto":
            cam = st.camera_input("Jepret Sampah")
            if cam:
                img = Image.open(cam)
                st.session_state['last_img'] = img
                st.session_state['last_mode'] = 'camera'

        elif mode == "📹 Live Video":
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

    # PANEL HASIL
    with col_kanan:
        st.markdown("#### 2. Hasil Analisis AI")
        
        if mode != "📹 Live Video":
            if 'last_img' in st.session_state and st.session_state.get('last_mode') in ['upload', 'camera']:
                img = st.session_state['last_img']
                lbl, conf = prediksi_gambar_diam(img)
                info = info_sampah[lbl]
                
                # Kartu Hasil
                st.markdown(f"""
                <div class="info-card">
                    <h2 style="margin-top:0; color:{'#2E7D32' if lbl=='ORGANIK' else '#E65100'}">{info['judul']}</h2>
                    <p style="font-size:1.1em;"><b>{info['sifat']}</b></p>
                    <hr>
                    <p>{info['desc']}</p>
                    <p>{info['bahaya']}</p>
                    <div style="background-color:#E0F2F1; padding:10px; border-radius:10px;">
                        <p style="margin:0;">{info['manfaat']}</p>
                    </div>
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
            
    # INFO TAMBAHAN DI BAWAH SCANNER (UNTUK HP)
    st.write("---")
    st.markdown("""
    <div style="background-color: #E0F7FA; padding: 15px; border-radius: 10px; border-left: 5px solid #00ACC1;">
        <h4>💡 Tips Lingkungan:</h4>
        <p style="margin:0;">Tahukah kamu? Satu liter minyak jelantah bisa mencemari 1000 liter air tanah! Jangan buang di wastafel ya.</p>
    </div>
    """, unsafe_allow_html=True)

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

# === TAB 3: KUIS PER LEVEL ===
with tab3:
    st.subheader("🎓 Tantangan Pemilahan Sampah")
    st.write("Pilih level kesulitanmu dan buktikan kamu peduli lingkungan!")
    
    st.divider()

    # Pilih Level
    pilihan_level = st.selectbox("Pilih Tingkat Kesulitan:", list(kuis_data.keys()))
    
    # Ambil soal berdasarkan level
    soal_aktif = kuis_data[pilihan_level]
    
    score = 0
    total_soal = len(soal_aktif)
    point_per_soal = 100 / total_soal
    
    for i, q in enumerate(soal_aktif):
        st.markdown(f"**Soal {i+1}:** {q['t']}")
        jawaban = st.radio(f"Jawaban No {i+1}:", q['o'], key=f"{pilihan_level}_{i}", index=None)
        
        if jawaban:
            if jawaban == q['k']:
                st.success(f"✅ {q['msg']}")
                score += point_per_soal
            else:
                st.error("❌ Masih kurang tepat, coba lagi ya.")
        st.write("---")
        
    if st.button("Lihat Nilai Saya"):
        if score == 100:
            if lottie_flower: st_lottie(lottie_flower, height=250, key="flower")
            st.markdown(f"### 🎉 SEMPURNA! Nilai: 100")
            st.success("Hebat! Kamu sudah siap jadi Duta Lingkungan.")
        elif score >= 60:
            st.markdown(f"### 👍 Bagus! Nilai: {int(score)}")
            st.info("Sedikit lagi sempurna. Ayo belajar lagi.")
        else:
            st.markdown(f"### Nilai: {int(score)}")
            st.warning("Jangan menyerah! Baca lagi info di menu Scanner ya.")

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
        - 🎓 Modul Kuis Berjenjang (Leveling)
        """)
        st.caption("Dibuat dengan ❤️ menggunakan Python & Streamlit")
