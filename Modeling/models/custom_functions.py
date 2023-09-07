# Regression imputation of missing values
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

def regression_imputation(data):
    data_copy = data.copy()

    cols_with_missing = [col for col in data_copy.columns if data_copy[col].isnull().any()]

    for col in cols_with_missing:
        data_copy[col + '_was_missing'] = data_copy[col].isnull()

    # Imputation
    my_imputer = IterativeImputer()
    imputed_data = pd.DataFrame(my_imputer.fit_transform(data_copy))
    imputed_data.columns = data_copy.columns

    return imputed_data


# KNN Imputer : CVBKNN (Cross Validation Based KNN Imputer)
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
import numpy as np

def cvbknn_imputer(X, n_splits, k_values=[], verbose=False):
    """
    X (np.array): data with missing values
    n_splits: number of splits for cross validation
    k_values: list of k values to try
    verbose: print progress

    Returns:
    data: (np.array) imputed data
    best_k: (int) best k value
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_k = None
    best_error = float('inf')

    for k in k_values:
        fold_errors = []
        for train_idx, test_idx in kf.split(X):
            # split data into train and test set for cross validation
            X_train, X_test = X[train_idx], X[test_idx]

            # create a copy of the test set and randomly set 10% of its known to NaN for imputation evaluation
            np.random.seed(42)
            X_test_masked = X_test.copy()
            mask = np.random.choice([True, False], size=X_test.shape, p=[0.1, 0.9])
            X_test_masked[mask] = np.nan

            # impute the missing values in the test set using KNN based on the train set
            imputer = KNNImputer(n_neighbors=k)
            imputer.fit(X_train)
            X_test_imputed = imputer.transform(X_test_masked)

            # Evaluate the imputation error (MSE)  for this fold
            mask_nan_removed = np.logical_and(mask, ~np.isnan(X_test))
            fold_error = mean_squared_error(X_test[mask_nan_removed], X_test_imputed[mask_nan_removed], squared=True)
            fold_errors.append(fold_error)

        # Calculate the average imputation error for this k value
        avg_error = np.mean(fold_errors)

        if verbose:
            print(f"Avg error for k={k}: {avg_error}")

        # update the best k value if this k value has a lower imputation error
        if avg_error < best_error:
            best_error = avg_error
            best_k = k

    # impute the missing values in the entire dataset using the best k value
    imputer = KNNImputer(n_neighbors=best_k)
    X_imputed = imputer.fit_transform(X)

    return X_imputed, best_k

def processing(X, n_splits=5, k_values=[1, 3, 5, 7, 9, 11], verbose=False):
    """
    Perform impuation on the data
    X: (np.array) data with missing values
    Returns:
    imputed_data_original_scale: imputed data with the original scale
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform CVBKNN imputation
    X_imputed_scaled, best_k = cvbknn_imputer(X_scaled, n_splits=n_splits, k_values=k_values, verbose=verbose)

    # Revert to the original scale
    imputed_data_original_scale = scaler.inverse_transform(X_imputed_scaled)

    # return imputed_data_original_scale as a dataframe
    return pd.DataFrame(imputed_data_original_scale)

#------------------------------------------------------------------------------------------

# Handling Class Imbalance
def handle_class_imbalance(dataset):
    # Splitting the dataset into features and target variable
    X = dataset.drop(columns=['Frailty_State'], axis=1)
    y = dataset['Frailty_State']

    # Combining them into a single DataFrame
    combined = pd.concat([X, y], axis=1)

    # Separating the classes
    df_majority = combined[combined.Frailty_State == 0]
    df_minority = combined[combined.Frailty_State == 1]

    # Oversampling the Minority Class
    df_minority_oversampled = resample(df_minority, 
                                       replace=True, 
                                       n_samples=len(df_majority), 
                                       random_state=42)

    # Combining the majority class DataFrame with the oversampled minority class DataFrame
    df_oversampled = pd.concat([df_majority, df_minority_oversampled])

    # Undersampling the Majority Class
    df_majority_undersampled = resample(df_majority, 
                                        replace=False, 
                                        n_samples=len(df_minority), 
                                        random_state=42)

    # Combining the minority class DataFrame with the undersampled majority class DataFrame
    df_undersampled = pd.concat([df_majority_undersampled, df_minority])

    return df_oversampled, df_undersampled

#------------------------------------------------------------------------------------------

# Splitting the data into train and test sets
def split_data(df):
  X = df.drop(['Frailty_State'], axis=1).values
  y = df['Frailty_State'].values

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  return X_train, X_test, y_train, y_test