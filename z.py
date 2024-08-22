from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Load the pre-trained model
model_path = 'CNN77.h5'
model = load_model(model_path)

# Define category dictionary
category_dict = {0: 'Puppy', 1: 'Senior'}

# Preprocess the image
def preprocess_image(image: np.ndarray) -> np.ndarray:
    resized_img = cv2.resize(image, (100, 100))
    grayscale_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    normalized_img = grayscale_img / 255.0
    preprocessed_img = normalized_img.reshape(1, 100, 100, 1)  # Add batch dimension and channel dimension
    return preprocessed_img

# Endpoint for image classification
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image file")

        # Preprocess the image
        preprocessed_img = preprocess_image(img)

        # Predict the class of the image
        prediction = model.predict(preprocessed_img)
        label = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Prepare the response
        response = {
            "category": category_dict[label],
            "confidence": float(confidence)
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


