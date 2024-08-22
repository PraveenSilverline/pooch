from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Define the FastAPI app
app = FastAPI()

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

# Load the SVM model
model_path = r"D:\PoochPaw\Dataset\Function 01\Recovery date model API\svm_model.pkl"
svm_model = joblib.load(model_path)

# Define input data schema using Pydantic
class InputData(BaseModel):
    Disease_Type: int
    Treatment: int
    Age_Years: int
    Breed: int
    Sentiment: int
    Doctor_Rating: int

# Define label mappings
label_mappings = {
    'Disease Type': {0: 'Bacterial Dermatosis', 1: 'Fungal Infections', 2: 'Hypersensitivity Allergic'},
    'Treatment': {0: 'Anti-allergy medication', 1: 'Antibiotics', 2: 'Antifungal medication'},
    'Breed': {0: 'Beagle', 1: 'Bulldog', 2: 'German Shepherd', 3: 'Labrador', 4: 'Poodle'},
    'Sentiment': {0: 'Negative', 1: 'Neutral', 2: 'Positive'},
    'Recovery Time (Weeks)': {0: '1 week', 1: '2 weeks', 2: '3 weeks', 3: '4 weeks'}
}

# Endpoint to predict recovery time
@app.post("/predict_recovery_time")
def predict_recovery_time(data: InputData):
    # Convert input data into DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict using the loaded SVM model
    prediction = svm_model.predict(input_df)[0]
    recovery_time = label_mappings['Recovery Time (Weeks)'][prediction]

    return {"Predicted Recovery Time": recovery_time}

# Endpoint to get label mappings
@app.get("/label_mappings")
def get_label_mappings():
    return label_mappings

# Run the FastAPI application with uvicorn
# Use command: uvicorn main:app --reload to run the app and auto-reload on code changes
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

