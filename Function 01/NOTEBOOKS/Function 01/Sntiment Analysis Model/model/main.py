import numpy as np
import pandas as pd
import re
import string
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nltk.stem import PorterStemmer
import os

app = FastAPI()

# Path to the stopwords file (adjust as per your setup)
stopwords_path = os.path.join(r"C:\Users\Pasidhu\Desktop\Dogsentiment_analysis\need\corpora\stopwords\english")

# Read the stopwords
with open(stopwords_path, 'r') as file:
    sw = file.read().splitlines()

# Load the model
with open(r'C:\Users\Pasidhu\Desktop\Final poochpaw\Function 01\Sntiment Analysis Model\model/model1.pickle', 'rb') as f:
    model = pickle.load(f)

# Load vocabulary
vocab = pd.read_csv(r"C:\Users\Pasidhu\Desktop\Dogsentiment_analysis\need\vocabulary.txt", header=None)
tokens = vocab[0].tolist()

# Define the PorterStemmer
ps = PorterStemmer()

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def preprocessing(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)  # Remove URLs
    text = remove_punctuations(text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = ' '.join([word for word in text.split() if word not in sw])  # Remove stopwords
    text = ' '.join([ps.stem(word) for word in text.split()])  # Stemming
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
    
    if prediction == 0:
        sentiment = 'negative'
    else:
        sentiment = 'positive'
    
    return {"review_text": review.text, "sentiment": sentiment}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
