import joblib
import pandas as pd
from preprocess import clean_text           # Imported custom text cleaning function
from sklearn.metrics import classification_report
from train import X_test, y_test            # Test data imported from train.py

# Load the saved ML model
model = joblib.load("models/model.pkl")
# Load the saved TF-IDF vectorizer
tfidf = joblib.load("models/vectorizer.pkl")

# Convert X_test text data into numeric vectors using TF-IDF
X_test_tfidf = tfidf.transform(X_test)

# Make predictions on test data
y_pred = model.predict(X_test_tfidf)        # Model predicts labels for test data

# Print evaluation metrics like precision, recall, f1-score
print(classification_report(y_test, y_pred))
