import json
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# Initialize the label encoder and scaler
le = LabelEncoder()
sc = StandardScaler()

# Load the dataset and model
df = pd.read_csv("./BPM_values.csv")
model = load_model('heart_anomaly_model.h5')

# Encode categorical variables
df['gender'] = le.fit_transform(df['gender'])
df['bred'] = le.fit_transform(df['bred'])
df['State'] = le.fit_transform(df['State'])
state_name = le.classes_

# Split data into training and testing sets
X = df.drop(columns=['State'])
y = df['State']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=69)

# Initialize FastAPI app
app = FastAPI()

# Define input model to accept a list of inputs
class TextIn(BaseModel):
    input_list: list

# Define output model for predictions
class PredictionOut(BaseModel):
    output: str

@app.get("/")
def home():
    return {"API_health_check": "OK", "model_version": "0.1.0"}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    try:
        # Create mapping dictionaries
        gender_map = {"female": 0, "male": 1}
        breed_map = {"German Shepherd": 0, "Bulldog": 1, "Labrador Retriever": 2, "Golden Retriever": 3}

        # Convert the payload input_list to a numpy array
        input_data = payload.input_list

        # Apply mappings to the input data
        input_data[1] = gender_map[input_data[1]]
        input_data[3] = breed_map[input_data[3]]

        # Convert input_data to a numpy array
        X_new = np.array([input_data])

        # Scale the data
        x_train_scaled = sc.fit_transform(x_train)
        X_new_scaled = sc.transform(X_new)

        # Make prediction
        prediction = model.predict(X_new_scaled)
        prediction = np.argmax(prediction, axis=-1)

        # Return the predicted state name
        return {"output": state_name[prediction][0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8002)
