import streamlit as st
import joblib
import numpy as np

# Mengimport logika RAG dari file sebelah
from rag_logic import dapatkan_sapaan_awal, tanggapi_chat_lanjutan

# ==========================================
# 1. KONFIGURASI API & MODEL ML
# ==========================================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
NAMA_FILE_PDF = "Guidebook_Stunting.pdf"  # Pastikan file PDF ini ada di folder PECUT AI

# Membaca model ML (disesuaikan dengan file stuning kamu)
model_ml = joblib.load("Nutri_RF.pkl")

# Tampilan UI Aplikasi Web
st.set_page_config(page_title="PecutStunt-AI Arsitektur Terpisah", layout="centered")
st.title("👶 PecutStunt-AI: Chatbot Hibrida (Modular)")
st.write(
    "UI bersih dengan pemisahan file logika RAG (rag_logic.py) & Machine Learning."
)

# Form Input Data Anak
jk = st.selectbox("Jenis Kelamin Balita", ["Laki-laki", "Perempuan"])
umur = st.number_input("Umur (Bulan)", min_value=0, max_value=60, value=12)
tinggi = st.number_input(
    "Tinggi Badan (cm)", min_value=30.0, max_value=120.0, value=75.0
)
berat = st.number_input("Berat Badan (kg)", min_value=1.0, max_value=30.0, value=9.0)

jk_bin = 0 if jk == "Laki-laki" else 1

# Inisialisasi Memori Chatbot (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "status_deteksi" not in st.session_state:
    st.session_state.status_deteksi = None

# Tombol Eksekusi Utama
if st.button("Mulai Diagnosis & Aktifkan Dokter AI", type="primary"):
    # 1. Jalankan Prediksi ML
    data_pasien = np.array([[jk_bin, umur, tinggi, berat]])
    hasil_ml = model_ml.predict(data_pasien)
    st.session_state.status_deteksi = (
        "BERISIKO STUNTING" if hasil_ml[0] == 1 else "NORMAL / AMAN"
    )

    # 2. Panggil Fungsi Sapaan Awal dari file rag_logic.py
    with st.spinner("Menghubungkan ke server Dokter AI..."):
        jawaban_awal = dapatkan_sapaan_awal(
            GEMINI_API_KEY,
            NAMA_FILE_PDF,
            jk,
            umur,
            tinggi,
            berat,
            st.session_state.status_deteksi,
        )

    # Masukkan respons dokter ke dalam memori chat
    st.session_state.messages = [{"role": "assistant", "content": jawaban_awal}]

# ==========================================
# 2. INTERFACE CHATBOX INTERAKTIF
# ==========================================
if st.session_state.status_deteksi:
    st.write("---")
    st.subheader("💬 Ruang Konsultasi Lanjutan Bersama Dokter AI")

    if st.session_state.status_deteksi == "BERISIKO STUNTING":
        st.error(f"Status Deteksi Model ML: {st.session_state.status_deteksi}")
    else:
        st.success(f"Status Deteksi Model ML: {st.session_state.status_deteksi}")

    # Menampilkan seluruh riwayat chat di layar
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Menangani input chat baru dari user
    if user_input := st.chat_input(
        "Tanyakan hal mendetail terkait isi buku panduan..."
    ):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Dokter sedang menganalisis buku panduan..."):
                # Panggil Fungsi Chat Lanjutan dari file rag_logic.py
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
