from flask import Flask, render_template, request
import joblib
import os
from textblob import TextBlob
import numpy as np
from scipy.sparse import hstack


app = Flask(__name__)

# Load models and vectorizer
models = {
    'Logistic Regression': joblib.load('model/log_reg.pkl'),
    'Linear Regression': joblib.load('model/lin_reg.pkl'),
    'Naive Bayes': joblib.load('model/nb_model.pkl'),
    'XGBoost': joblib.load('model/xgb_model.pkl'),
    'SVM': joblib.load('model/svm_model.pkl'),
    'Decision Tree': joblib.load('model/dt_model.pkl')
}

vectorizer = joblib.load('model/vectorizer.pkl')

# Load and format accuracies
accuracies = joblib.load('model/accuracies.pkl')
accuracies = {k: round(v, 2) for k, v in accuracies.items()}

# Mapping prediction outputs to labels
sentiment_labels = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

def get_extra_features(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    # Assuming you used 6 features in training: polarity, subjectivity, neg, neu, pos, compound
    # If you didn’t include VADER in training, just use 0s for the last 4
    return np.array([[polarity, subjectivity, 0, 0, 0, 0]])  # Shape (1, 6)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    vect_input = vectorizer.transform([user_input])  # Shape (1, 10000)
    extra_features = get_extra_features(user_input)  # Shape (1, 6)
    full_input = hstack([vect_input, extra_features])  # Shape (1, 10006)

    results = {}
    for name, model in models.items():
        prediction = model.predict(full_input)[0]  # make sure full_input exists
        sentiment = sentiment_labels.get(prediction, "Unknown")
        results[name] = sentiment


    return render_template('result.html', user_input=user_input, predictions=results)

@app.route('/compare')
def compare():
    return render_template('compare.html', accuracies=accuracies)

@app.route('/links')
def links():
    useful_links = ["https://github.com/ParthKhiriya/Sentiment_Analysis_PRML_Project"]
    return render_template('links.html', links=useful_links)

if __name__ == '__main__':
    app.run(debug=True)
