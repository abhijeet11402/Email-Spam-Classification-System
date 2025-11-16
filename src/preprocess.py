import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load English stopwords into a set for fast checking
stop_words = set(stopwords.words('english'))
# Initialize Porter Stemmer for reducing words to their base form
# ps = PorterStemmer()

def clean_text(text):
    
    """
    Preprocess the input text by:
    - converting to lowercase
    - removing URLs
    - removing special characters
    - removing stopwords
    - applying stemming
    """
    
    text = text.lower()
    text = re.sub(r'http\S+','',text)
    # Remove all characters except alphabets and spaces
    text = re.sub(r'[^a-zA-Z ]','',text)
    words = text.split()
    # Remove stopwords and apply stemming to each remaining word
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)
