from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tensorflow as tf
import uvicorn
from io import BytesIO

# Load the saved model
model = tf.keras.models.load_model("dog.h5")

category_dict = {0: 'other', 1: 'dog'}

app = FastAPI()

# preprocess the image
def preprocess_image(image: np.ndarray) -> np.ndarray:
    test_img = cv2.resize(image, (100, 100))
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    test_img = test_img / 255.0
    test_img = np.expand_dims(test_img, axis=-1)  # Add channel dimension
    test_img = np.expand_dims(test_img, axis=0)   # Add batch dimension
    return test_img

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image")

        # Preprocess the image
        test_img = preprocess_image(image)

        # Make predictions
        results = model.predict(test_img)
        label = np.argmax(results, axis=1)[0]  
        acc = int(np.max(results, axis=1)[0] * 100)  

        
        return JSONResponse(content={
            "predicted_class": category_dict[label],
            "accuracy": acc
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
  uvicorn.run(app, host="127.0.0.1", port=8000)
