import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
from src.preprocess import clean_text

# Load dataset
df = pd.read_csv('data/spam.csv', encoding='cp1252')
# Used encdoing cp1252 because the original email Spam dataset contains special characters.

# Clean text using the preprocessing function
df['clean_text'] = df['text'].apply(clean_text)

# Convert categorical labels to numeric (ham=0, spam=1)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],   # processed text
    df['label_num'], 
    test_size=0.2,      # 20% test, 80% train
    random_state=42      # for reproducibility
)

# TF-IDF
tfidf = TfidfVectorizer(max_features=3000)      # limit vocabulary to 3000 words
X_train_tfidf = tfidf.fit_transform(X_train)    # fit + transform training data
X_test_tfidf = tfidf.transform(X_test)          # only transform test data

# Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save trained model and TF-IDF vectorizer
joblib.dump(model, 'models/model.pkl')
joblib.dump(tfidf, 'models/vectorizer.pkl')

print("Training completed!")
