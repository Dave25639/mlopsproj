"""
Streamlit frontend for Food Classification API.
"""

import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

# Optional: Uncomment to use Google Cloud Run auto-detection
# from google.cloud import run_v2


@st.cache_resource
def get_backend_url():
    """
    Get the URL of the backend service.

    First tries to get it from Google Cloud Run, then falls back to
    the BACKEND environment variable, and finally defaults to localhost.
    """
    # Optional: Uncomment to use Google Cloud Run auto-detection
    # try:
    #     project = os.environ.get("GCP_PROJECT", "<project>")
    #     region = os.environ.get("GCP_REGION", "<region>")
    #     parent = f"projects/{project}/locations/{region}"
    #     client = run_v2.ServicesClient()
    #     services = client.list_services(parent=parent)
    #     for service in services:
    #         if service.name.split("/")[-1] == "production-model":
    #             return service.uri
    # except Exception as e:
    #     st.warning(f"Could not get backend from Cloud Run: {e}")

    # Fall back to environment variable or localhost
    backend_url = os.environ.get("BACKEND", "http://localhost:8000")
    return backend_url.rstrip("/")


def predict_image(image: Image.Image, backend_url: str, top_k: int = 5):
    """
    Send image to backend API for prediction.

    Args:
        image: PIL Image object
        backend_url: URL of the backend API
        top_k: Number of top predictions to return

    Returns:
        dict: Prediction response from API
    """
    try:
        # Convert PIL Image to bytes
        img_bytes = BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        # Prepare the request
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        data = {"top_k": top_k}

        # Send request to backend
        response = requests.post(
            f"{backend_url}/predict/upload",
            files=files,
            data=data,
            timeout=30
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {str(e)}")
        st.info(f"Make sure the backend is running at: {backend_url}")
        return None
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Food Classifier",
        page_icon="🍕",
        layout="wide"
    )

    st.title("🍕 Food Classification App")
    st.markdown("Upload an image of food to classify it using our Vision Transformer model!")

    # Get backend URL
    backend_url = get_backend_url()

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.text_input("Backend URL", value=backend_url, disabled=True, help="Set BACKEND environment variable to change")

        # Test backend connection
        if st.button("Test Backend Connection"):
            try:
                response = requests.get(f"{backend_url}/health", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Backend is connected!")
                    health = response.json()
                    st.json(health)
                else:
                    st.error(f"Backend returned status {response.status_code}")
            except Exception as e:
                st.error(f"❌ Cannot connect to backend: {str(e)}")

        st.divider()
        top_k = st.slider("Number of predictions to show", min_value=1, max_value=10, value=5)

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image of food to classify"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Predict button
            if st.button("🔍 Predict", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    result = predict_image(image, backend_url, top_k=top_k)

                    if result:
                        # Store result in session state for display
                        st.session_state['prediction_result'] = result
                        st.session_state['uploaded_image'] = image
                        st.rerun()

    with col2:
        st.header("📊 Results")

        if 'prediction_result' in st.session_state and st.session_state['prediction_result']:
            result = st.session_state['prediction_result']

            # Display top prediction prominently
            st.markdown("### 🎯 Top Prediction")
            top_pred = result['top_prediction']
            top_conf = result['top_confidence']

            st.success(f"**{top_pred.replace('_', ' ').title()}**")
            st.progress(top_conf)
            st.caption(f"Confidence: {top_conf * 100:.2f}%")

            st.divider()

            # Display all predictions
            st.markdown(f"### Top {len(result['predictions'])} Predictions")

            for pred in result['predictions']:
                class_name = pred['class_name'].replace('_', ' ').title()
                confidence = pred['confidence']

                # Create a nice progress bar for each prediction
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"{pred['rank']}. {class_name}")
                with col_b:
                    st.write(f"{confidence * 100:.1f}%")

                st.progress(confidence)
        else:
            st.info("👆 Upload an image and click 'Predict' to see results here")

    # Footer
    st.divider()
    st.caption(f"Backend: {backend_url} | Model: Vision Transformer (ViT)")


if __name__ == "__main__":
    main()
