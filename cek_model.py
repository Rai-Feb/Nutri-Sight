import google.generativeai as genai

# Masukkan API Key kamu yang valid di sini
genai.configure(api_key="AIzaSyDpqrwhI4TMrhw0e941tgFAqqNSDEkk2OY")

# Menampilkan semua model yang mendukung generate content
print("=== DAFTAR MODEL GEMINI YANG TERSEDIA ===")
for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(f"- {model.name}")
