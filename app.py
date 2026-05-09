import streamlit as st
from PIL import Image
import tempfile

from image_model import predict_image

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Deepfake Image Detection",
    page_icon="🖼️"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🖼️ AI-Powered Deepfake Image Detection")

st.write(
    "Upload an image to detect REAL or DEEPFAKE"
)

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# IMAGE DISPLAY
# ---------------------------------------------------

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Image",
        width=300
    )

    # ---------------------------------------------------
    # ANALYZE BUTTON
    # ---------------------------------------------------

    if st.button("Analyze Image"):

        with st.spinner("Analyzing Image..."):

            # Save uploaded image temporarily
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            )

            # Write bytes correctly
            temp_file.write(uploaded_file.getvalue())

            temp_file.close()

            # Prediction
            result = predict_image(temp_file.name)

            # ---------------------------------------------------
            # RESULT
            # ---------------------------------------------------

            if result == "DEEPFAKE IMAGE":

                st.error(
                    "🚨 DEEPFAKE IMAGE DETECTED"
                )

            else:

                st.success(
                    "✅ REAL IMAGE DETECTED"
                )

        st.balloons()