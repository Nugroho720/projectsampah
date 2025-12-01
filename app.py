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
from streamlit_option_menu import option_menu
import requests
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="EcoSort Pro X",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS MODERN (GLASSMORPHISM STYLE) ---
st.markdown("""
<style>
    /* Import Font Keren */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    /* Background Gradient Halus */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Styling Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }

    /* Card Style untuk Info */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }

    /* Hasil Deteksi Card */
    .result-card {
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        color: white;
        margin-bottom: 20px;
    }
    .organic {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .anorganic {
        background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%);
    }

    /* Judul Halaman */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    
    /* Tombol Kustom */
    .stButton>button {
        border-radius: 50px;
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        padding: 10px 25px;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(56, 239, 125, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. DATABASE & ASET ---
info_sampah = {
    "ORGANIK": {
        "judul": "🌱 ORGANIK (Organic)",
        "sifat": "Mudah Terurai (Biodegradable)",
        "desc": "Sampah ini berasal dari alam dan bisa dijadikan pupuk kompos.",
        "contoh": "Sisa makanan, Daun, Kulit Buah, Kayu kecil."
    },
    "ANORGANIK": {
        "judul": "⚙️ ANORGANIK (Inorganic)",
        "sifat": "Sulit Terurai (Non-Biodegradable)",
        "desc": "Sampah buatan pabrik. Wajib didaur ulang agar tidak mencemari lingkungan.",
        "contoh": "Plastik, Botol, Kaleng, Kaca, Styrofoam."
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
lottie_eco = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_tutvdkg0.json")

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

            # Desain Kotak Video Lebih Modern
            cv2.rectangle(img, (0, 0), (img.shape[1], 60), (0, 0, 0), -1) 
            cv2.putText(img, f"DETEKSI: {label}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            
            # Garis bidik
            h, w, _ = img.shape
            cv2.line(img, (w//2-20, h//2), (w//2+20, h//2), (255, 255, 255), 2)
            cv2.line(img, (w//2, h//2-20), (w//2, h//2+20), (255, 255, 255), 2)
            
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

def tampilkan_hasil_keren(label, conf):
    style_class = "organic" if label == "ORGANIK" else "anorganic"
    data = info_sampah[label]
    
    st.markdown(f"""
    <div class="result-card {style_class}">
        <h1 style="color:white; margin:0;">{data['judul']}</h1>
        <p style="font-size:1.2em; opacity:0.9;">{data['sifat']}</p>
        <h3 style="color:white; margin-top:10px;">Akurasi AI: {conf*100:.1f}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📚 Pelajari lebih lanjut tentang sampah ini"):
        st.info(f"**Definisi:** {data['desc']}")
        st.success(f"**Contoh:** {data['contoh']}")

# ================= MAIN APP =================

# --- SIDEBAR MENU (INI YANG BIKIN MEWAH) ---
with st.sidebar:
    if lottie_eco: st_lottie(lottie_eco, height=150)
    st.write("---")
    
    # Menu Navigasi Keren
    selected = option_menu(
        menu_title="Main Menu",
        options=["Scanner", "Dashboard", "Kuis", "About"],
        icons=["camera-fill", "bar-chart-fill", "controller", "info-circle-fill"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#ffffff"},
            "icon": {"color": "green", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#4CAF50"},
        }
    )
    
    st.write("---")
    st.caption("© 2025 Project UAS | v2.0 Pro")

if model is None:
    st.error("⚠️ Model AI tidak ditemukan! Pastikan file 'model_sampah.h5' ada di GitHub.")
    st.stop()

# --- HALAMAN 1: SCANNER ---
if selected == "Scanner":
    st.title("📸 AI Waste Scanner")
    st.write("Pilih metode input untuk mendeteksi jenis sampah.")
    
    # Custom Tab Style
    input_type = option_menu(
        menu_title=None,
        options=["Upload File", "Kamera", "Live Video"],
        icons=["cloud-upload", "camera", "webcam"],
        orientation="horizontal",
    )
    
    st.write("---")

    if input_type == "Upload File":
        c1, c2 = st.columns([1, 1])
        with c1:
            uploaded = st.file_uploader("📂 Tarik gambar sampah kesini", type=['jpg','png','jpeg'])
            if uploaded:
                img = Image.open(uploaded)
                st.image(img, use_container_width=True, caption="Preview Gambar")
        with c2:
            if uploaded:
                with st.spinner("Sedang menganalisis..."):
                    lbl, conf = prediksi_gambar_diam(img)
                    tampilkan_hasil_keren(lbl, conf)
                    if st.button("💾 Simpan Data Log"):
                        st.session_state['history_data'].append({"Waktu": datetime.now().strftime("%H:%M"), "Jenis": lbl})
                        st.toast("Data berhasil disimpan!", icon="✅")

    elif input_type == "Kamera":
        col_cam, col_res = st.columns([1, 1])
        with col_cam:
            cam = st.camera_input("Jepret Foto")
        with col_res:
            if cam:
                img = Image.open(cam)
                lbl, conf = prediksi_gambar_diam(img)
                st.image(img, width=300)
                tampilkan_hasil_keren(lbl, conf)
                if st.button("💾 Simpan Hasil"):
                    st.session_state['history_data'].append({"Waktu": datetime.now().strftime("%H:%M"), "Jenis": lbl})
                    st.toast("Tersimpan!")

    elif input_type == "Live Video":
        st.info("💡 Arahkan sampah ke kotak deteksi di bawah ini.")
        col_vid, col_info = st.columns([2, 1])
        with col_vid:
            webrtc_streamer(key="live", mode=WebRtcMode.SENDRECV,
                            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                            video_processor_factory=VideoProcessor,
                            media_stream_constraints={"video": True, "audio": False},
                            async_processing=True)
        with col_info:
            st.markdown("""
            ### Petunjuk:
            1. Pastikan cahaya cukup terang.
            2. Dekatkan sampah ke kamera.
            3. Tunggu hingga kotak deteksi muncul.
            """)

# --- HALAMAN 2: DASHBOARD ---
elif selected == "Dashboard":
    st.title("📊 Statistik Sampah")
    st.markdown("Pantau data sampah yang telah kamu scan secara real-time.")
    
    if st.session_state['history_data']:
        df = pd.DataFrame(st.session_state['history_data'])
        
        # Metrik Cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Scan", len(df), "sampah")
        m2.metric("Organik", len(df[df['Jenis']=='ORGANIK']), "item", delta_color="normal")
        m3.metric("Anorganik", len(df[df['Jenis']=='ANORGANIK']), "item", delta_color="inverse")
        
        st.write("---")
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig = px.pie(df, names='Jenis', hole=0.5, 
                         color='Jenis', color_discrete_map={'ORGANIK':'#38ef7d', 'ANORGANIK':'#ff5e62'},
                         title="Persentase Sampah")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("### 📝 Log Aktivitas")
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("Belum ada data. Yuk scan sampah dulu di menu Scanner!")

# --- HALAMAN 3: KUIS ---
elif selected == "Kuis":
    st.title("🎮 Kuis Lingkungan")
    st.write("Seberapa jago kamu memilah sampah?")
    
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        q1 = st.radio("1. Botol kaca bekas sirup termasuk jenis sampah apa?", ["Organik", "Anorganik", "Residu"], index=None)
        if q1:
            if q1 == "Anorganik": st.success("Benar! Kaca sulit terurai.")
            else: st.error("Salah. Kaca adalah anorganik.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.write("")
    
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        q2 = st.radio("2. Manakah yang bisa dijadikan pupuk kompos?", ["Plastik", "Kulit Pisang", "Kaleng"], index=None)
        if q2:
            if q2 == "Kulit Pisang": st.success("Tepat sekali! 🍌")
            else: st.error("Kurang tepat.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- HALAMAN 4: ABOUT ---
elif selected == "About":
    st.title("ℹ️ Tentang Aplikasi")
    c1, c2 = st.columns([1, 2])
    with c1:
        if os.path.exists("grafik_performa.png"):
            st.image("grafik_performa.png", caption="Grafik Akurasi Model", use_container_width=True)
    with c2:
        st.markdown("""
        ### EcoSort Pro X
        Aplikasi ini dikembangkan sebagai tugas akhir mata kuliah **Pengenalan Pola**.
        Menggunakan teknologi **Deep Learning (CNN)** untuk mengklasifikasikan citra sampah secara otomatis.
        
        **Teknologi:**
        - Python & TensorFlow
        - Streamlit Framework
        - OpenCV Computer Vision
        """)
        st.info("Developed with ❤️ by Kelompok 5")
