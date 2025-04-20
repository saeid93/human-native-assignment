# evaluate.py
from sklearn.metrics import classification_report

def evaluate(model, X_test, y_test, model_type="tfidf"):
    print(f"\n📊 Evaluation: {model_type.upper()}")

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["No PII", "PII"])
    print(report)
