from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class CustomImblearnBalancedBagging(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, n_estimators=10, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.balanced_bagging = BalancedBaggingClassifier(estimator=self.estimator,
                                                          n_estimators=self.n_estimators,
                                                          random_state=self.random_state)

    def fit(self, X, y):
        self.balanced_bagging.fit(X, y)
        return self

    def predict(self, X):
        return self.balanced_bagging.predict(X)

    def predict_proba(self, X):
        return self.balanced_bagging.predict_proba(X)

    def get_metrics(self, y_true, y_pred):
        idx = ~np.isnan(y_true)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "F1_score": f1_score(y_true, y_pred),
            "ROC_AUC_score": roc_auc_score(y_true, y_pred)
        }

        print(f'Confusion Matrix:\n {confusion_matrix(y_true, y_pred)}')
        print(f'Accuracy: {metrics["accuracy"]}')
        print(f'Precision: {metrics["precision"]}')
        print(f'Recall: {metrics["recall"]}')
        print(f'F1 Score: {metrics["F1_score"]}')
        print(f'ROC AUC Score: {metrics["ROC_AUC_score"]}')
        print('='*30)

        return metrics
