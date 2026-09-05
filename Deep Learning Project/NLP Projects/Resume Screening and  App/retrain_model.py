import pandas as pd
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv('Resume Screening.csv')

# Clean resume function (same as in notebook)
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
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

# Convert to array if needed
X_train = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
X_test = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

# Train the model (SVC as used in notebook)
svc_model = OneVsRestClassifier(SVC())
svc_model.fit(X_train, y_train)

# Evaluate
y_pred = svc_model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save the model artifacts
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(svc_model, open('clf.pkl', 'wb'))
pickle.dump(le, open('encoder.pkl', 'wb'))

print("Model artifacts saved successfully!")
print(f"TF-IDF vectorizer saved: tfidf.pkl")
print(f"Classifier model saved: clf.pkl")
print(f"Label encoder saved: encoder.pkl")