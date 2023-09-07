import pandas as pd
import os
import json
import numpy as np
from stabilogram.stato import Stabilogram
from descriptors import compute_all_features

#----------------------------------------------
# Function to load the gait data
def load_gait_data(foldername):
    base_path_gait = "./raw_data/{}/t0/gait/"

    gait_data_files = {
        'ce_data': "ACQ_CE.txt",
        'pd_data': "ACQ_PD.txt",
        'pg_data': "ACQ_PG.txt",
        'te_data': "ACQ_TE.txt"
    }

    try:
        data_gait = {}
        for name, filename in gait_data_files.items():
            path = base_path_gait.format(foldername) + filename
            data_gait[name] = pd.read_csv(path, sep="\t", skiprows=4)
        
        return data_gait
    except Exception as e:
        print("Gait data are not available")
        print(f"Error: {e}")  # Print the actual error message

#----------------------------------------------
# Function to load the posture data
def load_posture_data(foldername):
    base_path_posture = "P:/DATA_OCT_22/Expert_Eye/Dataset/gait_posture/raw_data/{}/t0/posture/"
    posture_data_files = {
        'yf_data': "2017-09-21_08_22_12_YF.txt", # Closed eyes
        'yo_data': "2017-09-21_08_22_12_YO.txt"  # Open eyes
    }
    try:
        data_posture = {}
        for name, filename in posture_data_files.items():
            path = base_path_posture.format(foldername) + filename
            data_posture[name] = pd.read_csv(path, sep="\t")

        return data_posture
    except:
        return None

#----------------------------------------------

# Function to load the gait features in the Features directory as json
def load_gait_features(foldername):
    base_path_gait = f"./raw_data/{foldername}/t0/gait/Features"
    gait_features_file = os.path.join(base_path_gait, "featXsens.json")

    try:
        with open(gait_features_file, 'r') as file:
            gait_feat = json.load(file)
            gait_features = pd.DataFrame(gait_feat['ListVariables'])
            gait_features[['Value1', 'Value2']] = pd.DataFrame(gait_features['Values'].to_list(), index=gait_features.index)
            gait_features = gait_features.drop('Values', axis=1)
            gait_features['Foldername'] = foldername  # add foldername as a column
            return gait_features
    except:
        # Return None if the file can't be found or opened
        return None  

#------------------------------------------------

# Create a function to create a dataframe with the gait features
def compute_gait_features():
    # Get the list of folders in the raw_data directory
    rootDir = 'p:/DATA_OCT_22/Expert_Eye/Dataset/gait_posture/raw_data'
    folder_names = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, name))]

    # Create a dataframe with the gait features for each subject
    gait_features_df = pd.DataFrame()
    for foldername in folder_names:
        gait_features = load_gait_features(foldername)
        if gait_features is not None:
            gait_features_df = pd.concat([gait_features_df, gait_features], ignore_index=True)
        else:
            # If there are no features, create a row with NaN for the Variable and Value1
            new_row = pd.DataFrame({'Foldername': [foldername], 'Variable': [np.nan], 'Value1': [np.nan]})
            gait_features_df = pd.concat([gait_features_df, new_row], ignore_index=True)

    # Pivot the dataframe
    wide_gait_features_df = gait_features_df.pivot(index='Foldername', columns='Variable', values='Value1')

    # Reset the index to make Foldername a column
    wide_gait_features_df.reset_index(level=0, inplace=True)

    return wide_gait_features_df


#------------------------------------------------
# Load posture features
def load_posture_features(foldername):
    base_path_posture = f"./raw_data/{foldername}/t0/posture/Features"
    posture_features_file = os.path.join(base_path_posture, "feat2017-09-21_08_22_12.json")

    try:
        with open(posture_features_file, 'r') as file:
            posture_feat = json.load(file)
            posture_features = pd.DataFrame(posture_feat['ListVariables'])
            posture_features[['Value1', 'Value2']] = pd.DataFrame(posture_features['Values'].to_list(), index=posture_features.index)
            posture_features = posture_features.drop('Values', axis=1)

        return posture_features
    except Exception as e:
        print("Posture features data are not available")
        print(f"Error: {e}")  # Print the actual error message

#------------------------------------------------

# function to create a dataframe with the posture features
def compute_posture_features(folder_names, skip_folders=[]):
    # Dataframe to store all the features
    all_features_df = pd.DataFrame()

    # Loop over all the folder names
    for foldername in folder_names:
        # If the foldername is in the skip_folders list, skip this iteration
        if foldername in skip_folders:
            print(f"Skipping folder: {foldername}")
            continue
        try:
            print(f"Processing folder: {foldername}")

            rawdata = load_posture_data(foldername)
            # If data is not loaded or if the data frames are empty,
            # add a row of NaNs to the DataFrame for this folder
            if rawdata is None or rawdata['yf_data'].empty or rawdata['yo_data'].empty:
                print(f"Folder '{foldername}' contains empty data files. Adding NaNs to DataFrame.")
                current_df = pd.DataFrame(index=[0])
                current_df['Foldername'] = foldername
                all_features_df = pd.concat([all_features_df, current_df], ignore_index=True)
                continue

            # Length and width of Nintendo wii balance board
            length = 53.6
            width = 33.7

            # The conditions identifiers
            eye_conditions = ['YF', 'YO']

            # A dictionary to store features for each eye condition
            features_dict = {}

            for i, condition in enumerate(eye_conditions):
                eye = rawdata[f'{condition.lower()}_data']

                # Calculate the total force and COP
                eye['TotalForce'] = (eye['BottomLeftCalcul_SensorsKG'] +
                    eye['BottomRightCalcul_SensorsKG'] +
                    eye['TopLeftCalcul_SensorsKG'] +
                    eye['TopRightCalcul_SensorsKG'])

                # Calculate COP_X and COP_Y
                eye['COP_X'] = ((eye['BottomLeftCalcul_SensorsKG'] +
                    eye['TopLeftCalcul_SensorsKG']) * width / 2 -
                    (eye['BottomRightCalcul_SensorsKG'] +
                    eye['TopRightCalcul_SensorsKG']) * width / 2) / eye['TotalForce']

                eye['COP_Y'] = ((eye['BottomLeftCalcul_SensorsKG'] +
                    eye['BottomRightCalcul_SensorsKG']) * length / 2 -
                    (eye['TopLeftCalcul_SensorsKG'] +
                    eye['TopRightCalcul_SensorsKG']) * length / 2) / eye['TotalForce']
                
                # Calculate the mean value of COP_X and COP_Y
                mean_COP_X = eye['COP_X'].mean()
                mean_COP_Y = eye['COP_Y'].mean()

                # Subtract the mean from each measurement to center the trajectories
                eye['COP_X_centered'] = eye['COP_X'] - mean_COP_X
                eye['COP_Y_centered'] = eye['COP_Y'] - mean_COP_Y

                time = eye['TIMESTAMP'].to_numpy()
                X = eye['COP_X_centered'].to_numpy()
                Y = eye['COP_Y_centered'].to_numpy()

                data = np.array([time, X, Y]).T

                # Verif if NaN data
                valid_index = (np.sum(np.isnan(data),axis=1) == 0)

                if np.sum(valid_index) != len(data):
                    raise ValueError("Clean NaN values first")

                stato = Stabilogram()
                stato.from_array(array=data)

                sway_density_radius = 0.3 # 3 mm

                params_dic = {"sway_density_radius": sway_density_radius}

                features = compute_all_features(stato, params_dic=params_dic)
                
                # Add the condition identifier to each key in the features dictionary
                features = {f'{k}_{condition}': v for k, v in features.items()}

                # Add the condition identifier to each key in the features dictionary
                features = {f'{k}_{condition}': v for k, v in features.items()}

                # Store the features for this condition in the features_dict
                features_dict[condition] = features

            # Combine features from all conditions into a single DataFrame
            all_features = pd.DataFrame({**features_dict['YF'], **features_dict['YO']}, index=[0])
            
            # Add the 'Foldername' to the all_features dataframe
            all_features['Foldername'] = foldername

            # Append the all_features dataframe to all_features_df
            all_features_df = pd.concat([all_features_df, all_features], ignore_index=True)

        except Exception as e:
            print(f"An error occurred while processing folder: {foldername}")
            print(f"Error: {e}")
            # optional: if you want to stop at the first error
            break

    # return the all_features_df
    return all_features_df

#------------------------------------------------


