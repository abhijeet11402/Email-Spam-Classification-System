## Overview
This project is a Spam/Ham Email classifier built with Python.
It uses TF-IDF vectorization and a Multinomial Naive Bayes model to predict whether a message is spam or not.
The project also provides a FastAPI backend to make predictions via a REST API.

## Features
Preprocess text: lowercase, remove URLs, special characters, stopwords, and apply stemming
Train a Naive Bayes classifier on SMS messages
Save trained model and TF-IDF vectorizer for later use
FastAPI endpoint to predict spam/ham in real-time.

## Installation

1. Clone the repository:
git clone <your-repo-link>
cd spam-detection

2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

3. Install dependencies:
pip install -r requirements.txt

## Training the Model
The script will load the dataset, clean text, vectorize, train a Naive Bayes model, and save the model and vectorizer in models/.
1. python src/train.py

## Running the FastAPI App
uvicorn main:app --reload
Swagger UI for testing: http://127.0.0.1:8000/docs
