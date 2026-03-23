import pandas as pd
import string
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# download stopwords
nltk.download('stopwords')

from nltk.corpus import stopwords

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("dataset.csv")

# ----------------------------
# Text Preprocessing
# ----------------------------

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["text"] = df["text"].apply(clean_text)

# ----------------------------
# Convert Text to Vectors
# ----------------------------

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

y = df["sentiment"]

# ----------------------------
# Train Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Train Model
# ----------------------------

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ----------------------------
# Prediction
# ----------------------------

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\nModel Training Completed")
print("------------------------")
print("Accuracy:", round(accuracy, 2))

print("\nClassification Report")
print("---------------------")
print(classification_report(y_test, pred))

# ----------------------------
# User Input Prediction
# ----------------------------

while True:

    print("\nEnter a sentence to analyze sentiment")
    print("Type 'exit' to stop")

    text = input("Input: ")

    if text.lower() == "exit":
        break

    text_clean = clean_text(text)

    vec = vectorizer.transform([text_clean])

    result = model.predict(vec)

    print("\nPrediction Result")
    print("-----------------")
    print("Sentence:", text)
    print("Sentiment:", result[0])

print("\nProgram Ended")