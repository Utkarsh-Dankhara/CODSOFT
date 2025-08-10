import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

# Getting the directory 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model, vectorizer, and encoder from the script's directory
model = joblib.load(os.path.join(BASE_DIR, "genre_classifier.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Example usage
if __name__ == "__main__":
    plot = input("Enter movie plot: ")
    processed = preprocess_text(plot)
    vec = vectorizer.transform([processed])
    pred_label = encoder.inverse_transform(model.predict(vec))[0]
    print(f"The Genre is, {pred_label}")
