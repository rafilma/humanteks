import re
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🤖",
    layout="centered"
)

# --- 2. FUNGSI PREPROCESSING (Sesuai persis dengan fix.ipynb) ---
def text_preprocessing(text):
    # 1. cleaning
    text = text.replace('-', ' ')
    text = re.sub(r'[\r\xa0\t]', '', text)
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'\b\w*\.com\w*\b', '', text)
    text = re.sub(r'\[.*?\]|\(.*?\\}|\{.*?\}', '', text)
    text = re.sub(r'\b(\w+)/(\w+)\b', r'\1 atau \2', text)
    text = re.sub(r'@[A-Za-z0-9]+|#[A-Za-z0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ')
    text = text.strip(' ')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 2. case folding
    text = text.lower()
    return text

# --- 3. LOAD MODEL & TOKENIZER (Menggunakan Cache agar Cepat) ---
@st.cache_resource
def load_assets():
    # Membaca file Tokenizer sebagai string mentah (.read())
    with open('tokenizer.json', 'r') as f:
        tokenizer_json = f.read() 
    tokenizer = tokenizer_from_json(tokenizer_json)
        
    # Memuat model dengan format Keras (.keras)
    model = tf.keras.models.load_model('model_bilstm.keras')
    return tokenizer, model

try:
    tokenizer, model = load_assets()
except Exception as e:
    st.error(f"Gagal memuat model/tokenizer. Pastikan file 'model_bilstm.keras' dan 'tokenizer.json' berada di folder yang sama. Error: {e}")
    st.stop()

# --- 4. TAMPILAN ANTARMUKA (UI) ---
st.title("🤖 AI vs Human Text Detector")
st.write("Masukkan teks atau artikel di bawah ini untuk mendeteksi apakah teks tersebut ditulis oleh manusia atau dihasilkan oleh AI.")

# Input Teks area
user_input = st.text_area("Masukkan Teks Di Sini:", height=250, placeholder="Ketik atau tempel teks teks Anda...")

# Parameter maxlen saat pelatihan model (Sesuaikan dengan notebook Anda)
MAX_LEN = 200 

if st.button("Deteksi Teks", type="primary"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu!")
    else:
        with st.spinner("Sedang menganalisis teks..."):
            # 1. Preprocessing teks input
            cleaned_text = text_preprocessing(user_input)
            
            # 2. Tokenizing & Padding
            sequences = tokenizer.texts_to_sequences([cleaned_text])
            padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
            
            # Konversi tipe data ke float32 untuk kestabilan prediksi TensorFlow
            padded = np.array(padded, dtype=np.float32)
            
            # 3. Prediksi Model
            prediction = model.predict(padded)[0][0]
            
            # 4. Menampilkan Hasil
            st.subheader("Hasil Analisis:")
            
            # Berdasarkan Confusion Matrix pada notebook Anda: xticklabels=['Human (0)', 'AI (1)']
            # Label 0 = Human, Label 1 = AI (Artinya mendekati 1 adalah AI)
            if prediction >= 0.5:
                confidence = prediction * 100
                st.error(f"🚨 **Terdeteksi sebagai teks buetan AI** ({confidence:.2f}% kepastian)")
            else:
                confidence = (1 - prediction) * 100
                st.success(f"✍️ **Terdeteksi sebagai teks buatan Manusia** ({confidence:.2f}% kepastian)")
                
            # Fitur Opsional untuk Debugging
            with st.expander("Lihat Detail Teknis (Debugging)"):
                st.write("**Teks Hasil Preprocessing:**", cleaned_text)
                st.write("**Hasil Representasi Angka (Sequences):**", sequences)
                st.write("**Nilai Probabilitas Mentah Model (Sigmoid):**", float(prediction))
