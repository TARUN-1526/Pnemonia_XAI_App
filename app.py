import streamlit as st
import tensorflow as tf
import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from PIL import Image
import gdown
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Pneumonia Detection using XAI",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# DOWNLOAD MODELS
# =========================================================
MODELS = {

    "lung_segmentation_model.pth":
    "https://drive.google.com/uc?id=1ohLgdge1YkM3xo2dXDJobe-zc2laLdYO",

    "vgg16_pneumonia_bg_removed112.h5":
    "https://drive.google.com/uc?id=1b-jMY926_n5z8ozSLoxAINxBo3L7NjSN",

    "vgg16_with_background.h5":
    "https://drive.google.com/uc?id=1mJHIhCaRAbZ65kMq70WzGPGTb2L5Ip6q",

    "resnet50_with_background.h5":
    "https://drive.google.com/uc?id=1_pHFm4F8Flfme3ScMYDDkoNBgH4V4KgS"
}

# =========================================================
# DOWNLOAD IF NOT EXISTS
# =========================================================
for filename, url in MODELS.items():

    if not os.path.exists(filename):

        with st.spinner(f"Downloading {filename}..."):

            gdown.download(
                url,
                filename,
                quiet=False
            )

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #eef4ff, #dbeafe);
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #0F172A;
}

.sub-title {
    text-align: center;
    font-size: 20px;
    color: #475569;
    margin-bottom: 30px;
}

.card {
    background: rgba(255,255,255,0.97);
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.prediction-normal {
    color: #16A34A;
    font-size: 18px;
    font-weight: bold;
}

.prediction-pneumonia {
    color: #DC2626;
    font-size: 18px;
    font-weight: bold;
}

.footer {
    text-align:center;
    color: gray;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class='main-title'>
🩺 Pneumonia Detection using Explainable AI
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='sub-title'>
U-Net Lung Segmentation + VGG16 + ResNet50 Comparative Analysis
</div>
""", unsafe_allow_html=True)

st.write("")

# =========================================================
# LOAD SEGMENTATION MODEL
# =========================================================
@st.cache_resource
def load_segmentation_model():

    device = torch.device("cpu")

    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    model.load_state_dict(
        torch.load(
            "lung_segmentation_model.pth",
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model

segmentation_model = load_segmentation_model()

# =========================================================
# LOAD CLASSIFICATION MODELS
# =========================================================
@st.cache_resource
def load_models():

    model_vgg_bg = tf.keras.models.load_model(
        "vgg16_with_background.h5"
    )

    model_vgg_seg = tf.keras.models.load_model(
        "vgg16_pneumonia_bg_removed112.h5"
    )

    model_resnet_bg = tf.keras.models.load_model(
        "resnet50_with_background.h5"
    )

    return (
        model_vgg_bg,
        model_vgg_seg,
        model_resnet_bg
    )

(
    model_vgg_bg,
    model_vgg_seg,
    model_resnet_bg
) = load_models()

# =========================================================
# PREPROCESS IMAGE
# =========================================================
def preprocess_image(image):

    img = np.array(image)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )

    img = cv2.resize(
        img,
        (224, 224)
    )

    img = img / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    return img

# =========================================================
# SEGMENT LUNGS
# =========================================================
def segment_lungs(image):

    image_rgb = np.array(image)

    img_resized = cv2.resize(
        image_rgb,
        (256, 256)
    )

    img_tensor = torch.tensor(
        img_resized / 255.0
    ).permute(2,0,1).float().unsqueeze(0)

    with torch.no_grad():

        pred = segmentation_model(img_tensor)

    mask = torch.sigmoid(
        pred
    ).squeeze().cpu().numpy()

    mask = (mask > 0.5).astype(np.uint8)

    mask = cv2.resize(
        mask,
        (
            image_rgb.shape[1],
            image_rgb.shape[0]
        )
    )

    lung_only = image_rgb * mask[:,:,None]

    segmented_image = Image.fromarray(
        lung_only.astype(np.uint8)
    )

    return segmented_image

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_image(model, processed_image):

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    probability = prediction[0][0]

    if probability > 0.5:

        predicted_class = "PNEUMONIA"

        confidence = probability * 100

    else:

        predicted_class = "NORMAL"

        confidence = (1 - probability) * 100

    return predicted_class, confidence

# =========================================================
# GRAD-CAM
# =========================================================
def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name
):

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(
                last_conv_layer_name
            ).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array
        )

        if isinstance(predictions, list):
            predictions = predictions[0]

        class_channel = predictions[:, 0]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        tf.multiply(
            pooled_grads,
            conv_outputs
        ),
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_val = tf.reduce_max(
        heatmap
    )

    if max_val != 0:
        heatmap /= max_val

    return heatmap.numpy()

# =========================================================
# OVERLAY HEATMAP
# =========================================================
def overlay_heatmap(
    heatmap,
    original_image,
    alpha=0.4
):

    img = np.array(original_image)

    heatmap = cv2.resize(
        heatmap,
        (
            img.shape[1],
            img.shape[0]
        )
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed_img = cv2.addWeighted(
        img,
        0.6,
        heatmap,
        alpha,
        0
    )

    return superimposed_img

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.title("🧠 AI Healthcare Dashboard")

    st.info("""
    ✔ U-Net Lung Segmentation  
    ✔ VGG16 Classification  
    ✔ ResNet50 Classification  
    ✔ Explainable AI (Grad-CAM)  
    ✔ Comparative Analysis  
    """)

    st.success(
        "✅ All Models Loaded Successfully"
    )

# =========================================================
# FILE UPLOADER
# =========================================================
uploaded_files = st.file_uploader(
    "📤 Upload Chest X-Ray Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =========================================================
# MAIN PROCESSING
# =========================================================
if uploaded_files is not None and len(uploaded_files) > 0:

    for uploaded_file in uploaded_files:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        segmented_image = segment_lungs(
            image
        )

        processed_original = preprocess_image(
            image
        )

        processed_segmented = preprocess_image(
            segmented_image
        )

        # =================================================
        # FILE TITLE
        # =================================================
        st.markdown(f"""
        <div class='card'>
        <h3 style='text-align:center;color:#0F172A;'>
        📁 {uploaded_file.name}
        </h3>
        </div>
        """, unsafe_allow_html=True)

        # =================================================
        # IMAGE DISPLAY
        # =================================================
        img1, img2 = st.columns(2)

        with img1:

            st.markdown(
                "<div class='card'>",
                unsafe_allow_html=True
            )

            st.subheader(
                "Original X-Ray"
            )

            st.image(
                image,
                width=300
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        with img2:

            st.markdown(
                "<div class='card'>",
                unsafe_allow_html=True
            )

            st.subheader(
                "Segmented Lung"
            )

            st.image(
                segmented_image,
                width=300
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        st.write("")

        # =================================================
        # MODEL RESULTS
        # =================================================
        col1, col2, col3 = st.columns(3)

        model_details = [

            (
                "VGG16 With BG",
                model_vgg_bg,
                processed_original,
                image,
                "block5_conv3"
            ),

            (
                "ResNet50 With BG",
                model_resnet_bg,
                processed_original,
                image,
                "conv5_block3_out"
            ),

            (
                "VGG16 Segmented",
                model_vgg_seg,
                processed_segmented,
                segmented_image,
                "block5_conv3"
            )
        ]

        columns = [
            col1,
            col2,
            col3
        ]

        for col, details in zip(
            columns,
            model_details
        ):

            (
                title,
                model,
                processed_img,
                display_img,
                layer
            ) = details

            with col:

                st.markdown(
                    "<div class='card'>",
                    unsafe_allow_html=True
                )

                st.subheader(title)

                pred, conf = predict_image(
                    model,
                    processed_img
                )

                # =========================================
                # DISPLAY PREDICTION
                # =========================================
                if pred == "PNEUMONIA":

                    st.markdown(
                        "<div class='prediction-pneumonia'>⚠ Pneumonia</div>",
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        "<div class='prediction-normal'>✅ Normal</div>",
                        unsafe_allow_html=True
                    )

                st.metric(
                    "Confidence",
                    f"{conf:.2f}%"
                )

                st.progress(
                    int(conf)
                )

                # =========================================
                # GRAD-CAM
                # =========================================
                heatmap = make_gradcam_heatmap(
                    processed_img,
                    model,
                    layer
                )

                gradcam = overlay_heatmap(
                    heatmap,
                    display_img
                )

                st.image(
                    gradcam,
                    caption="Grad-CAM",
                    width=220
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

        st.write("")
        st.write("---")

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class='footer'>
<hr>
<h4>🩺 AI Powered Pneumonia Detection System</h4>
<p>
Developed using U-Net, VGG16, ResNet50 and Explainable AI
</p>
</div>
""", unsafe_allow_html=True)
