from functions import create_gait_features_df

# Get the 'Feature dataframe'
feature_dataframe = create_gait_features_df()
feature_dataframe = feature_dataframe.dropna(axis=1, how='all')  # Drop columns with all NaN values


