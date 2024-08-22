from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
import string
import pickle
from nltk.stem import PorterStemmer
import os
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
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Path to the stopwords file
stopwords_path = os.path.join(r"D:\PoochPaw\Dataset\Function 01\sentiment model API\need\corpora\stopwords\english")

# Read the stopwords
with open(stopwords_path, 'r') as file:
    sw = file.read().splitlines()

# Load the model
with open(r'D:\PoochPaw\Dataset\Function 01\sentiment model API\model1.pickle', 'rb') as f:
    model = pickle.load(f)

# Load vocabulary
vocab = pd.read_csv(r"D:\PoochPaw\Dataset\Function 01\sentiment model API\need\vocabulary1.txt", header=None)
tokens = vocab[0].tolist()

# Define the PorterStemmer
ps = PorterStemmer()

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def preprocessing(text):
    text = text.lower()  
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) 
    text = remove_punctuations(text)
    text = re.sub(r'\d+', '', text) 
    text = ' '.join([word for word in text.split() if word not in sw])
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def vectorizer(ds, vocabulary):
    vectorized_lst = []
    
    for sentence in ds:
        sentence_lst = np.zeros(len(vocabulary))
        
        for i, token in enumerate(vocabulary):
            if token in sentence.split():
                sentence_lst[i] = 1
        
        vectorized_lst.append(sentence_lst)
    
    return np.array(vectorized_lst)

def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    return prediction

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(review: Review):
    preprocessed_txt = preprocessing(review.text)
    vectorized_txt = vectorizer([preprocessed_txt], tokens)
    prediction = get_prediction(vectorized_txt)
    
    sentiment = 'negative' if prediction == 0 else 'positive'
    
    return {"review_text": review.text, "sentiment": sentiment}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
