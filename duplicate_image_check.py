from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import os
import tempfile

app = FastAPI()

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data

def get_image_feature_vector(img_path):
    img_data = preprocess_image(img_path)
    feature_vector = model.predict(img_data)
    return feature_vector

def compare_images(img_path1, img_path2, threshold=0.5):
    vec1 = get_image_feature_vector(img_path1)
    vec2 = get_image_feature_vector(img_path2)
    cosine_sim = np.dot(vec1, vec2.T) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cosine_sim > threshold

@app.post("/compare-images/")
async def compare_images_endpoint(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Create a temporary directory to save uploaded files
    with tempfile.TemporaryDirectory() as tmpdirname:
        img_path1 = os.path.join(tmpdirname, file1.filename)
        img_path2 = os.path.join(tmpdirname, file2.filename)
        
        with open(img_path1, "wb") as buffer:
            buffer.write(file1.file.read())
        with open(img_path2, "wb") as buffer:
            buffer.write(file2.file.read())
        
        # Compare images
        is_same = compare_images(img_path1, img_path2)

    return JSONResponse(content={"is_same": bool(is_same)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

