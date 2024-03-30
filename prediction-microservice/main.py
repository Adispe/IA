import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from fastapi.responses import StreamingResponse
from PIL import Image

MODEL_PATH = r"UNET_Skyscan.h5"

app = FastAPI()

def prediction(input, model):

    predictions = model.predict(input)

    predicted_class = np.argmax(predictions, axis=-1)[0]

    #Attention les index rgb sont inversé (rouge = 0 0 255)
    colors = np.array([
        [0, 0, 0],  # 'no_data' class 0
        [128, 128, 128],  # 'clouds' class 1
        [0, 0, 255],  # 'artificial' class 2
        [0, 255, 0],  # 'cultivated' class 3
        [255, 0, 0],  # 'broadleaf' class 4
        [0, 255, 255],  # 'coniferous' class 5
        [0, 165, 255],  # 'herbaceous' class 6
        [255, 255, 0],  # 'natural' class 7
        [255, 255, 255],  # 'snow' class 8
        [255, 255, 255],  # 'water' class 
        ], dtype=np.uint8)

    return colors[predicted_class]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    image_PIL = Image.fromarray(image)
    image_array = img_to_array(image_PIL)
    input_arr = np.array([image_array])

    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        return JSONResponse(content={"error": f"Erreur lors du chargement du modèle : {str(e)}"}, status_code=500)

    output_img = prediction(input_arr, model)

    _, img_encoded = cv2.imencode('.jpg', output_img)

    # Convert binary image data to base64
    img_base64 = base64.b64encode(img_encoded)
    # Convert base64 bytes to a string
    img_base64_str = img_base64.decode('utf-8')

    return {"image_data": img_base64_str}