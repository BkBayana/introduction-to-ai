import joblib
model = joblib.load('spam_classifier_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
message = input("Введите сообщение: ")
vector = vectorizer.transform([message])
prediction = model.predict(vector)
print("Результат:", "Spam" if prediction[0] == 1 else "Not Spam")
