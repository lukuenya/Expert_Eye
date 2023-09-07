# custom_decision_tree.py

from decision_tree_NaN import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

class CustomDecisionTree:

    def __init__(self, max_depth=None, min_samples_split=2, nans_go_right=True, loss='gini'):
        self.dt_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, nans_go_right=nans_go_right, loss=loss)

    def fit(self, X_train, y_train):
        self.dt_model.train(X_train, y_train)

    def predict(self, X):
        return self.dt_model.predict(X)

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
