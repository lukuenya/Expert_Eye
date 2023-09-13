# custom_xgboost.py

import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt


class CustomXGBoost:

    # def __init__(self):
        #self.xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

    def __init__(self, objective='binary:logistic', eval_metric='logloss', learning_rate=None, n_estimators=None, max_depth=None, booster='dart', scale_pos_weight=2.7, random_state=42):
        self.objective = objective
        self.eval_metric = eval_metric
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.booster = booster
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state

        self.xgb_model = xgb.XGBClassifier(
            objective=self.objective,
            eval_metric=self.eval_metric, 
            learning_rate=self.learning_rate, 
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth, 
            booster=self.booster, 
            scale_pos_weight=self.scale_pos_weight, random_state=self.random_state)


    def fit(self, X_train, y_train):
        self.xgb_model.fit(X_train, y_train)

    def predict(self, X):
        return self.xgb_model.predict(X)

    def predict_proba(self, X):
        return self.xgb_model.predict_proba(X)
    
    def get_feature_importances(self):
        return self.xgb_model.feature_importances_

    def plot_feature_importances(self, feature_names):
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_names)), self.get_feature_importances())
        plt.yticks(range(len(feature_names)), [feature_names[i] for i in range(len(feature_names))])
        plt.xlabel('Feature Importance')
        plt.show()
    
    def get_params(self, deep=True):
        return {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
        self.xgb_model = xgb.XGBClassifier(
            objective=self.objective,
            eval_metric=self.eval_metric, 
            learning_rate=self.learning_rate, 
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth, 
            booster=self.booster, 
            scale_pos_weight=self.scale_pos_weight, 
            random_state=self.random_state)
        
        return self


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
    
    def plot_feature_importances(self, feature_names):
        # Ensure the model is fitted before plotting
        if self.xgb_model is None:
            raise RuntimeError("You must train the model before plotting feature importances.")
        
        # Get feature importances
        importances = self.xgb_model.feature_importances_
        
        # Sort the feature importances in descending order and take the top 10
        indices = np.argsort(importances)[::-1][:20]
        
        # Rearrange feature names so they match the sorted feature importances
        names = [feature_names[i] for i in indices]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Create a bar chart
        plt.barh(range(20), importances[indices])
        
        # Tick labels for the x-axis
        plt.yticks(range(20), names, rotation=0)
        
        # Axis labels and title
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance in Custom XGBoost Model')
        
        # Show the plot
        plt.show()
