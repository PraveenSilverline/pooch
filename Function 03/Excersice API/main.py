# 1. Library imports
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
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

# 3. Load the trained model using joblib
model_path = "knn_model_function03.pkl"
model = joblib.load(model_path)

# Define the request model
class PredictionRequest(BaseModel):
    Breed: int
    Age_Months: int
    Weight_lb: float
    Gender: int
    Weight_Goal_max_lb: float

# 7. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted exercise duration
@app.post('/predict')
def predict_exercise(data: PredictionRequest):
    try:
        data_dict = data.dict()
        input_data = pd.DataFrame([[
            data_dict['Breed'],
            data_dict['Age_Months'],
            data_dict['Weight_lb'],
            data_dict['Gender'],
            data_dict['Weight_Goal_max_lb']
        ]], columns=["Breed", "Age (Months)", "Weight (lb)", "Gender", "Weight Goal max (lb)"])
        
        prediction = model.predict(input_data)
        
        # Interpret the prediction for dog exercise durations
        duration_mapping = {
            0: 30,
            1: 45,
            2: 60,
            3: 90
        }
        recommendations = {
            0: 'Short walks or supervised playtime with light activities (e.g., fetch, frisbee)',
            1: 'Moderate walks with brisk pace or active playtime with fetch, frisbee, or tug-of-war',
            2: 'Brisk walks or hikes, active playtime with agility training, running or cycling with a dog leash, or engaging in dog sports',
            3: 'Long hikes or backpacking adventures, engaging in high-intensity activities like running or cycling (with proper training and conditioning)'
        }
        
        duration = duration_mapping.get(prediction[0], 0)
        recommendation = recommendations.get(prediction[0], 'No recommendation available')
        
        return {
            'Recommended Exercise Duration (minutes)': duration,
            'Recommendation': recommendation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 8. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn main:app --reload
