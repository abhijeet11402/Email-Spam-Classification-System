from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from src.preprocess import clean_text

# Initialize FastAPI application
app = FastAPI()

# Load trained machine learning model and TF-IDF vectorizer at startup
# (Loaded once → improves performance)
model = joblib.load("models/model.pkl")
tfidf = joblib.load("models/vectorizer.pkl")

# Define request body structure for prediction endpoint
class Email(BaseModel):
    text: str       # User must provide message text

@app.post("/predict")
def predict(data: Email):
    """
    Predict whether the given text is Spam or Ham.
    Steps:
        1. Clean the input text
        2. Convert text to TF-IDF vector
        3. Pass the vector to the model
        4. Return prediction result
    """
    cleaned = clean_text(data.text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    # Convert numeric result to a readable label
    result = "Spam" if prediction == 1 else "Ham"
    return {"prediction": result}
