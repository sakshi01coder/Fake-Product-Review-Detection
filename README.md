# 🕵️‍♀️ Fake Product Review Detection using Supervised Machine Learning

This project aims to detect **fake product reviews** using **Natural Language Processing (NLP)** and **Supervised Machine Learning algorithms**. It leverages text-based features and evaluates multiple classifiers to identify deceptive or spam-like reviews from genuine ones.

## 🧠 Algorithms Used
- Support Vector Machine (SVM) ✅ *Best performing model*
- Random Forest
- XGBoost
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Decision Tree
- Stochastic Gradient Descent (SGD)

## 🛠️ Tech Stack
- Python 🐍
- Scikit-learn
- Pandas
- Numpy
- Matplotlib / Seaborn
- Natural Language Toolkit (NLTK)

## 📝 Features
- Preprocessing of textual data (tokenization, lemmatization, stop-word removal)
- TF-IDF Vectorization
- Comparative analysis of model performance (Accuracy, Precision, Recall, F1 Score, ROC-AUC)
- Real-time user review input testing

## 📊 Results
| Model        | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------------|----------|-----------|--------|----------|---------|
| **SVM**      | 85.6%    | 0.861     | 0.852  | 0.857    | 0.856   |
| XGBoost      | 83.2%    | 0.819     | 0.855  | 0.837    | 0.832   |
| Naive Bayes  | 83.4%    | 0.857     | 0.804  | 0.830    | 0.834   |
| Random Forest| 82.9%    | 0.860     | 0.788  | 0.823    | 0.829   |
| KNN          | 55.0%    | 0.856     | 0.127  | 0.222    | 0.552   |

## 🧪 Sample Flow
1. User inputs a product review
2. Preprocessing is applied (cleaning + TF-IDF)
3. Model predicts whether the review is **Fake** or **Genuine**

## 🧾 Dataset
A labeled dataset of product reviews (genuine/fake) was used. Each review includes:
- `Review_Text`
- `Label` (1 = Genuine, 0 = Fake)

> **Note**: Dataset is not included due to licensing. Please use public datasets or your own for replication.

## 🏗️ Project Setup

### ✅ Prerequisites
- Python 3.8+
- pip (Python package manager)

### 🔧 Setup Commands

```bash
# Clone the repo
git clone https://github.com/yourusername/FakeReviewDetection.git
cd FakeReviewDetection
