from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load models and vectorizer
models = {
    'Logistic Regression': joblib.load('model/log_reg.pkl'),
    'Linear Regression': joblib.load('model/lin_reg.pkl'),
    'Naive Bayes': joblib.load('model/nb_model.pkl'),
    'XGBoost': joblib.load('model/xgb_model.pkl')
}
vectorizer = joblib.load('model/vectorizer.pkl')

# Load accuracies
accuracies = joblib.load('model/accuracies.pkl')
# Convert to percentage
accuracies = {k: round(v * 100, 2) for k, v in accuracies.items()}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    vect_input = vectorizer.transform([user_input])

    results = {}
    for name, model in models.items():
        prediction = model.predict(vect_input)[0]
        results[name] = "Positive" if prediction == 1 else "Negative"

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
