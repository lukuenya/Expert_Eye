import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt


class CustomXGBoost:

    def __init__(self, task='classification', **kwargs):
        self.task = task
        if self.task == 'classification':
            self.xgb_model = xgb.XGBClassifier(**kwargs)
        elif self.task == 'regression':
            self.xgb_model = xgb.XGBRegressor(**kwargs)
        else:
            raise ValueError(
                "Invalid task. Choose either 'classification' or 'regression'.")

    def fit(self, X_train, y_train):
        self.xgb_model.fit(X_train, y_train)

    def predict(self, X):
        return self.xgb_model.predict(X)

    def predict_proba(self, X):
        if self.task == 'classification':
            return self.xgb_model.predict_proba(X)
        else:
            raise ValueError(
                "predict_proba is not applicable for regression tasks.")

    def get_feature_importances(self):
        return self.xgb_model.feature_importances_

    def get_params(self, deep=True):
        return self.xgb_model.get_params(deep)

    def set_params(self, **parameters):
        self.xgb_model.set_params(**parameters)
        return self

    def get_metrics(self, y_true, y_pred):

        if self.task == 'classification':
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "F1_score": f1_score(y_true, y_pred),
                "ROC_AUC_score": roc_auc_score(y_true, y_pred)
            }
            print(f'Accuracy: {metrics["accuracy"]}')
            print(f'Precision: {metrics["precision"]}')
            print(f'Recall: {metrics["recall"]}')
            print(f'F1 Score: {metrics["F1_score"]}')
            print(f'ROC AUC Score: {metrics["ROC_AUC_score"]}')

        elif self.task == 'regression':

            metrics = {
                "MSE": mean_squared_error(y_true, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "MAE": mean_absolute_error(y_true, y_pred)
            }

            #print(f'MSE: {metrics["MSE"]}')
            #print(f'RMSE: {metrics["RMSE"]}')
            #print(f'MAE: {metrics["MAE"]}')

        else:
            raise ValueError(
                "Invalid task. Choose either 'classification' or 'regression'.")
        return metrics

    def plot_feature_importances(self, feature_names):
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_names)), self.get_feature_importances())
        plt.yticks(range(len(feature_names)), [
                   feature_names[i] for i in range(len(feature_names))])
        plt.xlabel('Feature Importance')
        plt.show()
