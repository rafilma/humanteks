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
    # Load Tokenizer
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
        
    # Load Model TensorFlow/Keras
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

# Parameter padding (Sesuaikan dengan maxlen saat Anda melatih model, contoh: 200)
MAX_LEN = 200 

if st.button("Deteksi Teks", type="primary"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu!")
    else:
        with st.spinner("Sedang menganalisis teks..."):
            # 1. Preprocessing
            cleaned_text = text_preprocessing(user_input)
            
            # 2. Tokenizing & Padding
            sequences = tokenizer.texts_to_sequences([cleaned_text])
            padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post') # sesuaikan post/pre dengan notebook Anda
            
            # 3. Prediksi Model
            prediction = model.predict(padded)[0][0]
            
            # 4. Menampilkan Hasil
            st.subheader("Hasil Analisis:")
            
            # Asumsi output sigmoid (0: Human, 1: AI) sesuai dengan xticklabels di confusion matrix Anda
            if prediction >= 0.5:
                confidence = prediction * 100
                st.error(f"🚨 **Terdeteksi sebagai teks buatan AI** ({confidence:.2f}% kepastian)")
            else:
                confidence = (1 - prediction) * 100
                st.success(f"✍️ **Terdeteksi sebagai teks buatan Manusia** ({confidence:.2f}% kepastian)")
                
            # Opsional: Tampilkan hasil teks setelah dibersihkan
            with st.expander("Lihat Teks Hasil Preprocessing"):
                st.write(cleaned_text)