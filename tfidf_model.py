# tfidf_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class TFIDFClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression()

    def fit(self, X: pd.Series, y: pd.Series) -> None:
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)

    def predict(self, X: pd.Series) -> list:
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)

    def predict_proba(self, X: pd.Series) -> list:
        X_vec = self.vectorizer.transform(X)
        return self.model.predict_proba(X_vec)
