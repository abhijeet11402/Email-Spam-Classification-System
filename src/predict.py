# src/predict.py
import joblib
from src.preprocess import clean_text

# File paths for the saved model and TF-IDF vectorizer
MODEL_PATH = 'models/model.pkl'
VEC_PATH = 'models/vectorizer.pkl'

def predict_text(text: str) -> dict:
    """
    Returns a dict: {'prediction_num': 0/1, 'label': 'ham'/'spam'}
    """
    # Load the trained ML model and TF-IDF vectorizer used during training.
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VEC_PATH)

    # Clean the input text using preprocessing function.
    cleaned = clean_text(text)
    # Convert cleaned text to TF-IDF vector form.
    vec = tfidf.transform([cleaned])
    # Get prediction (model.predict returns array → extract value)
    pred = int(model.predict(vec)[0])
    # Return both numeric and text label
    return {'prediction_num': pred, 'label': 'spam' if pred == 1 else 'ham'}

if __name__ == '__main__':
    import argparse
    # Create CLI argument parser for --text input
    parser = argparse.ArgumentParser()
    # Command-line argument to accept the text to classify
    parser.add_argument('--text', type=str, required=True)
    args = parser.parse_args()

    # Print the prediction result when run as a script.
    print(predict_text(args.text))
