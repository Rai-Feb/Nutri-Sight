import streamlit as st
import joblib
import numpy as np
from rag_logic import dapatkan_sapaan_awal, tanggapi_chat_lanjutan

# Konfigurasi Jalur File Proyek
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
NAMA_FILE_PDF = "Datasets/Guidebook_Cegah Stunting.pdf"
NAMA_MODEL_ML = "Nutri_RF.pkl"

model_ml = joblib.load(NAMA_MODEL_ML)

# Injeksi Tema Kemenkes RI (CSS)
st.set_page_config(
    page_title="Nutri-Sight", layout="centered", page_icon=""
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #007A5E; font-family: 'Segoe UI', sans-serif; font-weight: 700; margin-bottom: 0px; }
    h3 { color: #333333; font-family: 'Segoe UI', sans-serif; }
    .stButton>button {
        background-color: #00A67E;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 16px;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #007A5E; color: white; border: none; }
    </style>
""",
    unsafe_allow_html=True,
)

# Tata Letak Header Komponen Utama
# Tata Letak Header Komponen Utama
col_logo, col_judul = st.columns([1, 4])
with col_logo:
    try:
        # st.image("Nutri-Sight.png", width=110) # <-- Hapus baris ini
        # Ganti dengan CSS agar gambar tajam dan tidak pecah
        st.markdown(
            '<img src="Nutri-Sight.png" style="max-height: 80px; width: auto; image-rendering: -webkit-optimize-contrast;">',
            unsafe_allow_html=True,
        )
    except:
        st.caption("⚠️ File Nutri-Sight.png belum masuk folder root")
with col_judul:
    st.title("Nutri-Sight")
    st.write(
        "Sistem Pemantauan Gizi & Deteksi Dini Stunting Terintegrasi Kementerian Kesehatan RI"
    )
st.write("---")

# Pengelompokan Form Menjadi Dua Kolom Sejajar
st.subheader("📋 Formulir Antropometri Balita")
kolom_kiri, kolom_kanan = st.columns(2)

with kolom_kiri:
    jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    umur = st.number_input("Umur (Bulan)", min_value=0, max_value=60, value=12)
with kolom_kanan:
    tinggi = st.number_input(
        "Tinggi Badan (cm)", min_value=30.0, max_value=120.0, value=75.0
    )
    berat = st.number_input(
        "Berat Badan (kg)", min_value=1.0, max_value=30.0, value=9.0
    )

jk_bin = 0 if jk == "Laki-laki" else 1

# Manajemen Memori Percakapan
if "messages" not in st.session_state:
    st.session_state.messages = []
if "status_deteksi" not in st.session_state:
    st.session_state.status_deteksi = None

# Eksekusi Utama Analisis
if st.button("Mulai Analisis Gizi & Konsultasi AI"):
    data_pasien = np.array([[jk_bin, umur, tinggi, berat]])
    hasil_ml = model_ml.predict(data_pasien)
    st.session_state.status_deteksi = (
        "BERISIKO STUNTING" if hasil_ml[0] == 1 else "NORMAL / AMAN"
    )

    with st.spinner("Mengekstrak berkas panduan medis Kemenkes..."):
        jawaban_awal = dapatkan_sapaan_awal(
            GEMINI_API_KEY,
            NAMA_FILE_PDF,
            jk,
            umur,
            tinggi,
            berat,
            st.session_state.status_deteksi,
        )
    st.session_state.messages = [{"role": "assistant", "content": jawaban_awal}]

# Penanganan Output Diagnosis & Fitur Chatbox
if st.session_state.status_deteksi:
    st.write("---")
    st.subheader("📊 Hasil Diagnosis & Rekomendasi Klinis")

    if st.session_state.status_deteksi == "BERISIKO STUNTING":
        st.error(f"🚨 STATUS DETEKSI SISTEM: {st.session_state.status_deteksi}")
    else:
        st.success(f"✅ STATUS DETEKSI SISTEM: {st.session_state.status_deteksi}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_input := st.chat_input(
        "Ketik di sini untuk konsultasi lanjutan bersama Dokter AI Kemenkes..."
    ):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Dokter AI sedang memverifikasi rujukan buku panduan..."):
                jawaban_lanjutan = tanggapi_chat_lanjutan(
                    GEMINI_API_KEY,
                    NAMA_FILE_PDF,
                    st.session_state.status_deteksi,
                    jk,
                    umur,
                    st.session_state.messages[:-1],
                    user_input,
                )
                st.write(jawaban_lanjutan)
        st.session_state.messages.append(
            {"role": "assistant", "content": jawaban_lanjutan}
        )
