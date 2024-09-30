from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io
import tensorflow as tf

app = FastAPI()

model = tf.keras.models.load_model('./models/smile_model.h5')

@app.post("/predict/")
async def predict_smile(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = image.resize((64, 64)) 
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    smiling = bool(np.argmax(prediction, axis=1)[0])
    return {"smiling": smiling}
