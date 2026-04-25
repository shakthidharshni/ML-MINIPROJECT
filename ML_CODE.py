import pandas as pd
import numpy as np
import re
import string
import tkinter as tk
from tkinter import messagebox

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# -------------------------------
# 1. Load datasets
# -------------------------------
fake = pd.read_csv("D:/shakthi/ML_MINI_PROJECT/Fake.csv") 
true = pd.read_csv("D:/shakthi/ML_MINI_PROJECT/True.csv")

fake["label"] = 0   # Fake
true["label"] = 1   # Real

data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

# -------------------------------
# 2. Text cleaning
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data["text"] = data["text"].apply(clean_text)

# -------------------------------
# 3. Split data
# -------------------------------
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -------------------------------
# 4. TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# -------------------------------
# 5. Train model
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# 6. Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 7. Graphs
# -------------------------------
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
disp.plot()
plt.title("Confusion Matrix")
plt.show()

accuracy = accuracy_score(y_test, y_pred)

plt.figure()
plt.bar(["Accuracy"], [accuracy])
plt.title("Model Accuracy")
plt.ylim(0, 1)
plt.show()

# -------------------------------
# 8. Prediction Function
# -------------------------------
def predict_news(news):
    news = clean_text(news)
    vect = vectorizer.transform([news])
    prediction = model.predict(vect)
    return "Real News" if prediction[0] == 1 else "Fake News"

# -------------------------------
# 9. GUI using Tkinter
# -------------------------------
def check_news():
    news = text_box.get("1.0", tk.END)
    
    if news.strip() == "":
        messagebox.showwarning("Warning", "Please enter some news text")
        return
    
    result = predict_news(news)
    result_label.config(text="Prediction: " + result)

root = tk.Tk()
root.title("Fake News Detector")
root.geometry("500x400")

title = tk.Label(root, text="Fake News Detection", font=("Arial", 16, "bold"))
title.pack(pady=10)

text_box = tk.Text(root, height=10, width=50)
text_box.pack(pady=10)

check_button = tk.Button(root, text="Check News", command=check_news)
check_button.pack(pady=5)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
