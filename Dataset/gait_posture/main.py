import pandas as pd
import os
import numpy as np
from functions import compute_gait_features, compute_posture_features


# Get the list of folders in the raw_data directory
rootDir = 'p:/DATA_OCT_22/Expert_Eye/Dataset/gait_posture/raw_data'
folder_names = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, name))]

# Folders to skip for in computer_psture_features()
skip_folders = ['FRA005GMS', 'FRA008TJL', 'FRA010DMA', 'HUC001HMR', 'LEG007TGL', 'LEG018BPC', 'LEG027HJO', 'LEG034KLG', 'LEG042HJO', 'LEG047VSI', 'LEG048VHI', 'LEG049FAL', 'LEG050LMN']


# Load the questionnaire dataframe from the `Questionnaire` directory
questionnaire = pd.read_excel('p:/DATA_OCT_22/Expert_Eye/Dataset/Questionnaire/encoded_questionnaire.xlsx')


def main(folder_names, questionnaire):

    # Calculate gait features
    gait_features = compute_gait_features()
    gait_features = gait_features.dropna(axis=1, how='all')  # Drop columns with all NaN values

    # Calculate posture features
    posture_features = compute_posture_features(folder_names, skip_folders)

    # Reset indices of the feature dataframes
    gait_features.reset_index(drop=True, inplace=True)
    posture_features.reset_index(drop=True, inplace=True)

    # Reset index of the questionnaire dataframe
    questionnaire.reset_index(drop=True, inplace=True)

    # Merge all the dataframes on 'Foldername', filling missing values with NaN
    merged_df = questionnaire.merge(gait_features, on='Foldername', how='left')
    merged_df = merged_df.merge(posture_features, on='Foldername', how='left')

    # Replace any infinities with NaN
    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return merged_df

# Run the main function
result = main(folder_names, questionnaire)

# Convert all the columns in the dataframe to type int except for the Foldername column
result = result.astype({col: 'int64' for col in result.columns if col != 'Foldername'})

print(result.info())
print(result.head())