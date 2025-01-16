import os
from xgboost_feature_extractor import extract_features_from_genre_directory
from feauture_saver import create_feature_file

def prepare_xgboost_features(training_dataset_path  ,training_features_xgboost_dest, validation_dataset_path, validation_features_xgboost):
  
    labels = os.listdir(training_dataset_path)

    # Loop through each genre/label in the training dataset
    for label in labels:
        fullPath = os.path.join(training_dataset_path, label)
        song_feature_map, classLabel = extract_features_from_genre_directory(fullPath)
    
        # Check if song_feature_map is not empty
        if song_feature_map:
            # Create the folder for the label if it doesn't exist
            label_feature_path = os.path.join(training_features_xgboost_dest, label)
            os.makedirs(label_feature_path, exist_ok=True)  # Create folder if not exists

            # Iterate over each song's feature dataframe
            for song_name, features_df in song_feature_map.items():
                # Create the feature file path
                feature_file_path = os.path.join(label_feature_path, f'{song_name}.csv')

                # Check if the feature file already exists, if so, skip it
                if not os.path.exists(feature_file_path):
                    # If the file doesn't exist, create the feature file
                    create_feature_file(label_feature_path, f'{song_name}', features_df)
                    print(f"Created feature file for {song_name} in {label_feature_path}")
                else:
                    print(f"Feature file for {song_name} already exists. Skipping.")

    # Validation Dataset (same logic as above)
    
    labels = os.listdir(validation_dataset_path)
    
    # Loop through each genre/label in the validation dataset
    for label in labels:
        fullPath = os.path.join(validation_dataset_path, label)
        song_feature_map, classLabel = extract_features_from_genre_directory(fullPath)
    
        # Check if song_feature_map is not empty
        if song_feature_map:
            # Create the folder for the label in the validation dataset if it doesn't exist
            label_feature_path = os.path.join(validation_features_xgboost, label)
            os.makedirs(label_feature_path, exist_ok=True)  # Create folder if not exists

            # Iterate over each song's feature dataframe
            for song_name, features_df in song_feature_map.items():
                # Create the feature file path
                feature_file_path = os.path.join(label_feature_path, f'{song_name}.csv')

                # Check if the feature file already exists, if so, skip it
                if not os.path.exists(feature_file_path):
                    # If the file doesn't exist, create the feature file
                    create_feature_file(label_feature_path, f'{song_name}', features_df)
                    print(f"Created feature file for {song_name} in {label_feature_path}")
                else:
                    print(f"Feature file for {song_name} already exists. Skipping.")
