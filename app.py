import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
from tensorflow import keras

# -------------------------
# Helper functions
# -------------------------
@st.cache_resource
def load_model(path="mnist_cnn.keras"):
    """
    Load a saved Keras model from disk. Cached so it won't reload on every rerun.
    """
    if not os.path.exists(path):
        st.warning(f"Model '{path}' not found. Place your trained file in this folder.")
        return None
    return keras.models.load_model(path)

def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to the 28x28 grayscale normalized array expected by the CNN:
    - grayscale
    - invert (so strokes are bright, background dark)
    - resize to 28x28
    - normalize to [0,1]
    - add channel dimension -> (1, 28, 28, 1)
    """
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

def predict_image(model, img_array: np.ndarray):
    """
    Predict probabilities and label for a preprocessed image array.
    """
    probs = model.predict(img_array, verbose=0)
    pred = int(np.argmax(probs, axis=1)[0])
    return pred, probs[0]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="MNIST Digit Predictor", layout="centered")
st.title("✍️ Handwritten Digit Predictor (MNIST) — Streamlit (.keras)")

st.markdown(
    """
Upload a handwritten digit image or draw one in the canvas and the app will predict the digit (0–9)
using your **CNN** model.

**Tip:** Make the digit thick and centered on a white background for best results.
"""
)

# Load model once
model = load_model("mnist_cnn.keras")

# Initialize session storage for the current PIL image
if "pil_img" not in st.session_state:
    st.session_state.pil_img = None

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input")

    # UNIQUE KEY for radio
    input_mode = st.radio(
        "Choose input method:",
        ("Upload image", "Draw (canvas)"),
        key="input_mode_radio"
    )

    pil_img = None

    if input_mode == "Upload image":
        # UNIQUE KEY for uploader
        uploaded_file = st.file_uploader(
            "Upload a digit image (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            key="uploader_main"
        )
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")

    else:
        # drawing canvas
        from streamlit_drawable_canvas import st_canvas

        st.write("Draw a digit (use your mouse). Then click **Predict** on the right.")
        # OPTIONAL: clear canvas button
        clear = st.button("Clear canvas", key="clear_canvas_btn")

        # UNIQUE KEY for canvas; include a changing key part if clearing
        canvas_result = st_canvas(
            fill_color="rgb(255, 255, 255)",  # background white
            stroke_width=18,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"draw_canvas_{int(clear)}",  # toggles key when 'clear' is pressed
        )

        if canvas_result.image_data is not None:
            # canvas returns an RGBA array
            im = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
            pil_img = im.convert("RGB")

    st.markdown("---")
    st.header("Model")
    if model is None:
        st.error("❌ Model not loaded. Put `mnist_cnn.keras` in this folder and refresh.")
    else:
        st.success("✅ Model loaded successfully.")
    st.caption("Train separately and save using: `model.save('mnist_cnn.keras')`")

with col2:
    st.header("Preview & Prediction")

    # Update session image if a new one was provided this run
    if pil_img is not None:
        st.session_state.pil_img = pil_img

    if st.session_state.pil_img is not None:
        preview = st.session_state.pil_img.copy()
        st.image(preview, caption="Input image / drawing", width=280)

        if model is not None:
            # UNIQUE KEY for predict button
            if st.button("Predict", key="predict_btn_main"):
                img_arr = preprocess_pil_image(preview)
                pred_label, probs = predict_image(model, img_arr)

                st.subheader(f"🧮 Predicted digit: **{pred_label}**")

                st.write("### Probabilities")
                fig, ax = plt.subplots()
                ax.bar(range(10), probs)
                ax.set_xlabel("Digit")
                ax.set_ylabel("Probability")
                ax.set_xticks(range(10))
                st.pyplot(fig)

                top3_idx = probs.argsort()[-3:][::-1]
                st.write("### Top 3 predictions")
                for idx in top3_idx:
                    st.write(f"**{idx}** — {probs[idx]*100:.2f}%")
        else:
            st.info("Model not loaded — please add the `.keras` file and refresh.")
    else:
        st.info("No image yet — upload or draw on the left side.")

st.markdown("---")
st.caption("Built with TensorFlow / Keras (.keras format) and Streamlit — unique keys added to avoid duplicate IDs.")
