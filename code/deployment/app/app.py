import streamlit as st
import requests
from PIL import Image
import io

st.title("Smile Detection App")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button('Predict'):
        # Convert image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        
        # Send the image to FastAPI for prediction
        response = requests.post("http://api:8000/predict/", files={"file": img_bytes.getvalue()})
        result = response.json()
        
        # Display the result
        if not(result["smiling"]):
            st.success("The person is smiling 😊")
        else:
            st.warning("The person is not smiling 😐")
