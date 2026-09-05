import pandas as pd
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the data
df = pd.read_csv(os.path.join(BASE_DIR, 'Resume Screening.csv'))

# Clean resume function (same as in notebook)
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

# Clean the resumes
df['Resume'] = df['Resume'].apply(cleanResume)

# Encode labels
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
required_text = tfidf.fit_transform(df['Resume'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(required_text, df['Category'], test_size=0.2, random_state=42)

# Train the model (SVC as used in notebook)
svc_model = OneVsRestClassifier(SVC())
svc_model.fit(X_train, y_train)

# Evaluate
y_pred = svc_model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save the model artifacts
with open(os.path.join(BASE_DIR, 'tfidf.pkl'), 'wb') as f:
    pickle.dump(tfidf, f)
with open(os.path.join(BASE_DIR, 'clf.pkl'), 'wb') as f:
    pickle.dump(svc_model, f)
with open(os.path.join(BASE_DIR, 'encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

print("Model artifacts saved successfully!")
print("TF-IDF vectorizer saved: tfidf.pkl")
print("Classifier model saved: clf.pkl")
print("Label encoder saved: encoder.pkl")