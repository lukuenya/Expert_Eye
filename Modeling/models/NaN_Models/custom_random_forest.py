import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from custom_decision_tree import CustomDecisionTree


class CustomRandomForest:

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=0.7, class_weight='balanced', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)

        if self.class_weight == 'balanced':
            unique_classes, counts = np.unique(y, return_counts=True)
            n_samples = len(y)
            n_classes = len(unique_classes)
            self.class_weight = {cls: n_samples / (n_classes * count) 
                                for cls, count in zip(unique_classes, counts)}

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            # Create a tree and fit it
            tree = CustomDecisionTree(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split, class_weight=self.class_weight, max_features=self.max_features)
            
            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.zeros((X.shape[0], len(self.trees)))

        for i, tree in enumerate(self.trees):
            tree_preds[:, i] = tree.predict(X)

        # Majority voting
        y_pred = np.apply_along_axis(lambda x: np.bincount(
            x.astype('int')).argmax(), axis=1, arr=tree_preds)

        return y_pred

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "random_state": self.random_state,
            "class_weight": self.class_weight
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
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
