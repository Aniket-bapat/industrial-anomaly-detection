
import streamlit as st
import tempfile
import os

from PIL import Image
from anomaly_detection import predict_image


# Page configuration
st.set_page_config(
    page_title="Industrial Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("Industrial Anomaly Detection System")

st.write(
    "Upload a bottle image to identify whether "
    "it is normal or defective."
)

st.divider()


# Image upload
uploaded_file = st.file_uploader(
    "Upload bottle image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:

        st.subheader("Detection Result")

        if st.button("Analyze Image"):

            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".png"
                ) as temp_file:

                    temp_path = temp_file.name
                    image.save(temp_path)

                with st.spinner("Analyzing image..."):

                    prediction, score, threshold = (
                        predict_image(temp_path)
                    )

                if prediction == "Normal":

                    st.success("NORMAL IMAGE")

                else:

                    st.error("DEFECTIVE IMAGE")

                st.metric(
                    "Anomaly Score",
                    f"{score:.4f}"
                )

                st.metric(
                    "Detection Threshold",
                    f"{threshold:.4f}"
                )

                st.write("Prediction:", prediction)

            except Exception as error:

                st.error(f"Analysis failed: {error}")

            finally:

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
