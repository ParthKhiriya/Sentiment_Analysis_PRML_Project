# Sentiment Analysis using Machine Learning

This project aims to build a sentiment classification system using various machine learning models such as Logistic Regression, Linear Regression, Naive Bayes, and XGBoost. The primary goal is to determine whether a given piece of text (like a tweet or a review) expresses a positive or negative sentiment.

## 🚀 Features

- Text preprocessing and vectorization using TF-IDF
- Implementation of four ML models
- Comparison of model accuracies
- Web application using Flask for real-time sentiment prediction
- Project Page with theoretical insights and useful links

## 📂 Project Structure
Sentiment_Analysis_PRML_Project/
│
├── model/                 → Contains all trained model files and vectorizer
│   ├── log_reg.pkl
│   ├── lin_reg.pkl
│   ├── nb_model.pkl
│   ├── xgb_model.pkl
|   ├── svm_model.pkl
|   ├── dt_model.pkl
│   └── vectorizer.pkl
│   ├── accuracies.pkl     → Pickled dictionary storing accuracy of all 
│
├── templates/             → HTML templates used by Flask for rendering pages
│   ├── index.html
│   ├── result.html
│   ├── compare.html
│   └── links.html
│
├── static/                → for static files like CSS or images
├── models
├── app.py                 → Main Flask app that handles routing and logic
├── Copy_of_Project.ipynb  → Jupyter notebook with full EDA, training, and results
└── README.md              → Project overview and documentation

## 🔑 Kaggle API Setup for Google Colab

Since the dataset was downloaded from Kaggle and used in the notebook, to run this project on **Google Colab**, every team member needed to configure access to Kaggle’s API. Here's how to set it up:

### 🔧 Steps:
1. **Login to your Kaggle account**  
   Go to: https://www.kaggle.com/account  

2. **Scroll down to 'API' section**  
   Click on **"Create New API Token"**

3. **A file named `kaggle.json` will be downloaded**  
   It contains your username and Kaggle API key.

4. **Upload the file to Colab**  
   Use the following code in the first cell of the notebook:

   ```python
   from google.colab import files
   files.upload()  # Upload kaggle.json

   
## 📊 Models Used

- **Logistic Regression**  
- **Linear Regression**  
- **Multinomial Naive Bayes**  
- **XGBoost Classifier**  
- **Support Vector Machine (SVM)**  
- **Random Forest Classifier**  
- **K-Nearest Neighbors (KNN)**

## 🧠 Theory

Sentiment analysis is a Natural Language Processing (NLP) task where the goal is to classify text based on the sentiment expressed. It is commonly used in social media monitoring, product review classification, and customer feedback analysis.

We used:
- **TF-IDF Vectorizer**: Converts text to numeric feature vectors.
- **Supervised Learning**: The models are trained on labeled data for binary classification (positive/negative).

## 💡 Accuracy Comparison

Each model's accuracy (in %), computed on the test dataset:

| Model                  | Accuracy (%)  |
|------------------------|---------------|
| Logistic Regression    | 83.5          |
| Linear Regression      | 19.65         |
| Naive Bayes            | 78.4          |
| XGBoost                | 80.69         |
| SVM                    | 84.05         |
| Random Forest          | 80.32         |
| K-Nearest Neighbors    | 69.41         |

## 🌐 Live Project Page

Check out the web interface: [Project Website](http://127.0.0.1:5000)  
GitHub Repo: [https://github.com/ParthKhiriya/Sentiment_Analysis_PRML_Project](https://github.com/ParthKhiriya/Sentiment_Analysis_PRML_Project)  
Dataset: [https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)

## 📚 Resources

- Scikit-learn  
- XGBoost  
- Flask  
- Pandas, NumPy  
- NLTK for preprocessing

## 🧑‍💻 Team

- Parth Khiriya (B23EE1051)  
- Devesh Labana (B23CS1015)  
- Vishrut Aditya Ratnoo (B23EE1102)  
- Shlok Agrawal (B23EE1096)  
- Nitish Gupta (B23CS1046)  
- Divya Kumar (B23EE1019)  
