import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

class CustomBaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, n_estimators=10, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.estimators_ = []
        
        for _ in range(self.n_estimators):
            # Bootstrap resampling
            X_sample, y_sample = resample(X, y)
            
            # Clone and fit the base estimator
            estimator = self._clone_and_fit(X_sample, y_sample)
            self.estimators_.append(estimator)
            
        return self

    def _clone_and_fit(self, X, y):
        estimator = self.base_estimator
        estimator.fit(X, y)
        return estimator

    def predict(self, X):
        # Generate predictions from each estimator
        predictions = np.zeros((X.shape[0], len(self.estimators_)))
        for i, estimator in enumerate(self.estimators_):
            predictions[:, i] = estimator.predict(X)
        
        # Aggregate predictions to generate final prediction
        y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions.astype(int))
        
        return y_pred

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
