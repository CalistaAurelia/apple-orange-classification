import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Klasifikasi Gambar Apple vs Orange",
    page_icon="🍎🍊",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #ff4b4b;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stCaptionContainer"] {
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL 
# ============================================
MODEL_PATH = "models/soal1_custom_cnn_v1_apple_orange.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Sesuai notebook: IMG_SIZE = (128, 128), class_names = ['apple', 'orange']
IMG_SIZE = (128, 128)
CLASS_NAMES = ["apple", "orange"]

# ============================================
# SIDEBAR — INFORMATION MODEL
# ============================================
with st.sidebar:
    st.header("Information Model")
    st.markdown("**Model:** Custom CNN (V1)")
    st.markdown("**Jenis Model:** Convolutional Neural Network")
    st.markdown("**Kelas yang dapat diprediksi:**")
    st.markdown("- Apple")
    st.markdown("- Orange")
    st.divider()
    st.caption(
        "Model versi awal (V1). "
        "Accuracy: 62.73% | F1 Score: 0.3881. "
        "Model ini masih menunjukkan gejala overfitting dan bias terhadap salah satu kelas."
    )

# ============================================
# MAIN CONTENT
# ============================================

# 1. Judul aplikasi
st.markdown(
    '<h1 style="color:#ff4b4b;">Klasifikasi Gambar Apple 🍎 vs Orange 🍊</h1>',
    unsafe_allow_html=True
)

# 2. Upload gambar
with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Upload gambar untuk diklasifikasikan", type=["jpg", "jpeg", "png"]
    )

if uploaded_file is not None:
    # 3. Preview gambar
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", width=350)

    predict_clicked = st.button("Deteksi", type="primary")

    if predict_clicked:

        # Preprocessing: resize saja.
        # PENTING: layer Rescaling(1./255) sudah ada DI DALAM arsitektur model
        # (lihat cell arsitektur CNN V1 di notebook), jadi di sini TIDAK boleh
        # dinormalisasi manual lagi (tidak dibagi 255) supaya tidak double-scaling.
        img_resized = img.resize(IMG_SIZE)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)  # shape jadi (1, 128, 128, 3)

        # Prediksi
        with st.spinner("Memprediksi..."):
            prediction = model.predict(img_array)

        # Output sigmoid (binary): prediction berbentuk [[nilai]]
        # urutan class_names dari notebook: ['apple', 'orange'] -> orange = label 1
        orange_prob = float(prediction[0][0])
        apple_prob = 1 - orange_prob

        predicted_class = CLASS_NAMES[1] if orange_prob > 0.5 else CLASS_NAMES[0]
        final_confidence = orange_prob if orange_prob > 0.5 else apple_prob

        # 4 & 5. Hasil Klasifikasi + Tingkat Kepercayaan
        with st.container(border=True):
            st.subheader("Hasil Klasifikasi")
            st.markdown(f"**Jenis Gambar:** {predicted_class.upper()}")
            st.markdown(f"**Tingkat Kepercayaan:** {final_confidence*100:.2f}%")

            # 6. Probabilitas masing-masing kelas
            st.markdown("**Probabilitas Kelas:**")

            st.markdown(f"""
            <div style="margin-top:10px;">
                <p style="margin-bottom:3px; font-weight:600;">Apple 🍎 — {apple_prob*100:.2f}%</p>
                <div style="background-color:#eee; border-radius:8px; height:22px;">
                    <div style="width:{apple_prob*100}%; background-color:#e63946; height:22px; border-radius:8px; transition: width 0.6s ease;"></div>
                </div>
                <p style="margin:12px 0 3px 0; font-weight:600;">Orange 🍊 — {orange_prob*100:.2f}%</p>
                <div style="background-color:#eee; border-radius:8px; height:22px; margin-bottom:15px;">
                    <div style="width:{orange_prob*100}%; background-color:#ff8c42; height:22px; border-radius:8px; transition: width 0.6s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
