import google.generativeai as genai
from pypdf import PdfReader


# Read File PDF
def muat_dokumen_pdf(nama_file_pdf):
    try:
        reader = PdfReader(nama_file_pdf)
        teks_lengkap = ""
        for page in reader.pages:
            teks_lengkap += page.extract_text() + "\n"
        return teks_lengkap
    except Exception as e:
        return f"Gagal membaca PDF: {e}. Menggunakan teks cadangan standar Kemenkes."


# Function Respons Awal Chatbot
def dapatkan_sapaan_awal(api_key, nama_pdf, jk, umur, tinggi, berat, status_ml):
    genai.configure(api_key=api_key)
    dokumen_rujukan = muat_dokumen_pdf(nama_pdf)

    prompt = f"""
    Kamu adalah dokter spesialis anak Kemenkes RI di aplikasi PecutStunt-AI.
    Data Pasien: JK={jk}, Umur={umur} bulan, TB={tinggi} cm, BB={berat} kg.
    Diagnosis ML: Anak dinyatakan {status_ml}.
    
    TUGAS ANDA:
    Gunakan dokumen rujukan resmi hasil ekstrak PDF ini untuk memberikan sapaan awal, kesimpulan kondisi anak, dan solusi pertolongan pertama yang sesuai petunjuk buku panduan:
    {dokumen_rujukan}
    
    Jawab dengan poin-poin tebal yang rapi dan mudah dimengerti orang tua.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text


# Next Chat
def tanggapi_chat_lanjutan(
    api_key, nama_pdf, status_ml, jk, umur, riwayat_chat, pertanyaan_baru
):
    genai.configure(api_key=api_key)
    dokumen_rujukan = muat_dokumen_pdf(nama_pdf)

    prompt = f"""
    Anda adalah dokter spesialis anak Kemenkes RI di aplikasi PecutStunt-AI.
    Data Pasien saat ini: {status_ml}, JK={jk}, Umur={umur} bulan.
    Dokumen Rujukan Medis (Hasil PDF): {dokumen_rujukan}
    
    Riwayat obrolan kita sebelumnya: {riwayat_chat}
    Pertanyaan baru dari user: {pertanyaan_baru}
    
    Jawablah pertanyaan tersebut dan pastikan isi jawabanmu WAJIB bersumber atau selaras dengan Dokumen Rujukan Medis (Hasil PDF) di atas.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text
