import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import joblib


data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'text']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})



X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vectors, y_train)

y_pred = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


joblib.dump(model, 'spam_classifier_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("Модель и векторизатор успешно сохранены!")

print("Classification Results:")
print(f" Accuracy: {accuracy * 100:.2f}%")
print(f" Confusion Matrix:\n{conf_matrix}")


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Values (0: Not Spam, 1: Spam)')
plt.ylabel('Predicted Values (0: Not Spam, 1: Spam)')
plt.title('Actual vs Predicted Values (Spam Classification)')
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.plot([0, 1], [0, 1], 'r--')
plt.show()
