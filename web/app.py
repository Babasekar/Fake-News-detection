from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('fake_and_real_news_dataset.csv')

# Data preprocessing
def preprocess_data(df):
    df = df.dropna()
    X = df['text']
    y = df['label']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train models
def train_models(X_train, X_test, y_train, y_test):
    # Feature extraction using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.75)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    X_train_tfidf.sort_indices()
    X_test_tfidf.sort_indices()

    # Define models
    models = [
        RandomForestClassifier(n_estimators=100),
        DecisionTreeClassifier(),
        LogisticRegression(max_iter=1000)
    ]
    model_names = ['Random Forest', 'Decision Tree', 'Logistic Regression']

    accuracies = []
    for model, name in zip(models, model_names):
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    return model_names, accuracies

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        # Preprocess the input text
        # Use the trained model to make predictions
        # Return the prediction result to the user
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model_names, accuracies = train_models(X_train, X_test, y_train, y_test)
    app.run(debug=True)
