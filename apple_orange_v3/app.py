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


# ============================================
# CUSTOM CSS
# Tampilan dibuat mengikuti app (2).py
# ============================================
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
MODEL_PATH = "models/soal1_mobilenetv2_transfer_apple_orange.h5"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ============================================
# KONFIGURASI MODEL
# Sesuai notebook:
# IMG_SIZE = (128, 128)
# class_names = ['apple', 'orange']
#
# Model MobileNetV2 memiliki:
# - input 128 x 128 x 3
# - data augmentation di dalam model
# - Rescaling(1./127.5, offset=-1) di dalam model
# - output sigmoid untuk binary classification
#
# Karena preprocessing Rescaling sudah ada di dalam model,
# gambar TIDAK dinormalisasi manual di app.
# ============================================
IMG_SIZE = (128, 128)
CLASS_NAMES = ["apple", "orange"]


# ============================================
# SIDEBAR — INFORMATION MODEL
# ============================================
with st.sidebar:
    st.header("Information Model")

    st.markdown("**Model:** MobileNetV2 (Transfer Learning)")
    st.markdown("**Jenis Model:** Convolutional Neural Network")
    st.markdown("**Kelas yang dapat diprediksi:**")
    st.markdown("- Apple")
    st.markdown("- Orange")

    st.divider()

    st.caption(
        "MobileNetV2 menggunakan transfer learning dengan "
        "pretrained ImageNet. Base model dibekukan (frozen) "
        "dan ditambahkan Global Average Pooling, Dense 128, "
        "Dropout 0.3, serta output sigmoid."
    )

    st.caption(
        "Test Accuracy: 96.36% | F1 Score: 96.67%. "
        "Model menunjukkan performa terbaik dibandingkan "
        "model Custom CNN pada notebook."
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
        "Upload gambar untuk diklasifikasikan",
        type=["jpg", "jpeg", "png"]
    )


if uploaded_file is not None:

    # 3. Preview gambar
    img = Image.open(uploaded_file).convert("RGB")
    st.image(
        img,
        caption="Gambar yang diupload",
        width=350
    )

    predict_clicked = st.button(
        "Deteksi",
        type="primary"
    )

    if predict_clicked:

        # ============================================
        # PREPROCESSING
        # ============================================
        # Sesuai notebook, model menerima input 128x128x3.
        #
        # Tidak dilakukan /255 atau mnv2_preprocess()
        # secara manual karena model yang disimpan sudah
        # memiliki layer:
        #
        # Rescaling(1./127.5, offset=-1)
        #
        # di dalam arsitekturnya.
        # ============================================
        img_resized = img.resize(IMG_SIZE)

        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        # ============================================
        # PREDIKSI
        # ============================================
        with st.spinner("Memprediksi..."):
            prediction = model.predict(
                img_array,
                verbose=0
            )

        # ============================================
        # OUTPUT SIGMOID
        # ============================================
        # Notebook menggunakan:
        # label 0 = apple
        # label 1 = orange
        #
        # Karena output sigmoid:
        # prediction > 0.5 -> orange
        # prediction <= 0.5 -> apple
        # ============================================
        orange_prob = float(prediction[0][0])
        apple_prob = 1.0 - orange_prob

        if orange_prob > 0.5:
            predicted_class = CLASS_NAMES[1]
            final_confidence = orange_prob
        else:
            predicted_class = CLASS_NAMES[0]
            final_confidence = apple_prob

        # ============================================
        # HASIL KLASIFIKASI
        # ============================================
        with st.container(border=True):

            st.subheader("Hasil Klasifikasi")

            st.markdown(
                f"**Jenis Gambar:** {predicted_class.upper()}"
            )

            st.markdown(
                f"**Tingkat Kepercayaan:** "
                f"{final_confidence * 100:.2f}%"
            )

            # ========================================
            # PROBABILITAS MASING-MASING KELAS
            # ========================================
            st.markdown("**Probabilitas Kelas:**")

            st.markdown(f"""
            <div style="margin-top:10px;">
                <p style="margin-bottom:3px; font-weight:600;">Apple 🍎 — {apple_prob*100:.2f}%</p>
                <div style="background-color:#eee; border-radius:8px; height:22px; margin-bottom:15px;">
                    <div style="width:{apple_prob*100}%; background-color:#e63946; height:22px; border-radius:8px; transition: width 0.6s ease;"></div>
                </div>
                <p style="margin:12px 0 3px 0; font-weight:600;">Orange 🍊 — {orange_prob*100:.2f}%</p>
                <div style="background-color:#eee; border-radius:8px; height:22px; margin-bottom:15px;">
                    <div style="width:{orange_prob*100}%; background-color:#ff8c42; height:22px; border-radius:8px; transition: width 0.6s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
