import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import joblib  # Import joblib to load the scaler
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# List of origins that should be allowed to make requests
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://192.168.8.100:8000",
    "http://127.0.0.1:8000",
    # Add more origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow all origins or specify particular origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the column names
column_names = ["ANeck_x", "ANeck_y", "ANeck_z", "GNeck_x", "GNeck_y", "GNeck_z"]

# Load the saved model
model = load_model(r"C:\Users\athth\OneDrive\Documents\GitHub\Pooch-Paw-ML\Final\Function02\full_model1.h5")

# Load the saved StandardScaler
scaler = joblib.load(r"C:\Users\athth\OneDrive\Documents\GitHub\Pooch-Paw-ML\Final\Function02\scaler.pkl")

# Define the input data model
class InputData(BaseModel):
    ANeck_x: float
    ANeck_y: float
    ANeck_z: float
    GNeck_x: float
    GNeck_y: float
    GNeck_z: float

@app.post("/predict")
def predict(data: list[InputData]):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([item.dict() for item in data], columns=column_names)
    
    # Scale the input data using the loaded scaler
    X_scaled = scaler.transform(input_data)  # Use transform instead of fit_transform
    
    # Reshape the scaled data for Conv1D input
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    # Make predictions
    predictions = model.predict(X_reshaped)
    
    # Get the class with the highest probability
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Return the predictions
    return {"predicted_classes": predicted_classes.tolist()}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


# Run the server
# port = 8002
