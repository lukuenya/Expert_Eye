import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

class CustomLightGBM:
    
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'objective': 'binary',
                'metric': 'binary_error',
                'boosting_type': 'gbdt'
            }
        else:
            self.params = params
        self.lgb_model = None

    def fit(self, X_train, y_train):
        d_train = lgb.Dataset(X_train, label=y_train)
        self.lgb_model = lgb.train(self.params, d_train)

    def predict(self, X):
        return np.round(self.lgb_model.predict(X))

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
