import os
import shutil
import random
import subprocess
import xgboost_preprocessing
import xgboost_main
import knn_preprocessing
import knn_main
import kagglehub

def download_and_extract_gtzan():
   
    path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")

    print(f"Archive is downloaded at : {path}")
    if not os.path.exists(path):
        print(F"failed to download from kaggle andradaolteanu/gtzan-dataset-music-genre-classification")

    # The dataset will be extracted into a folder called 'gtzan-dataset'
    input_dir = os.path.join(path, "Data", "genres_original")
   
    return os.path.abspath(input_dir) 


def split_files(input_dir, train_dir, val_dir, train_ratio=80):
   
   for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)

        if os.path.isdir(folder_path):
            train_folder_path = os.path.join(train_dir, folder_name)
            val_folder_path = os.path.join(val_dir, folder_name)

            os.makedirs(train_folder_path, exist_ok=True)
            os.makedirs(val_folder_path, exist_ok=True)

            files = os.listdir(folder_path)

            random.shuffle(files)

            split_index = int(len(files) * train_ratio / 100)

            train_files = files[:split_index]
            val_files = files[split_index:]

            for file in train_files:
                shutil.copy(os.path.join(folder_path, file), os.path.join(train_folder_path, file))

            for file in val_files:
                shutil.copy(os.path.join(folder_path, file), os.path.join(val_folder_path, file))

            print(f"Folder '{folder_name}' split into {len(train_files)} training and {len(val_files)} validation files.")


input_dir = download_and_extract_gtzan()

train_dir = 'training_dataset'                                  # numele directory ului unde se afla datele originale (fisiere audio)  de antrenare
val_dir = 'validation_dataset'                                  # numele directory ului unde se afla datele originale (fisiere audio)  de validare
training_xgboost_features = "training_features_xgboost"         # directory ul unde vor fi salvate feature urile pt xgboost de antrenare
validation_xgboost_features = "validation_features_xgboost"     # directory ul unde vor fi salvate feature urile pt xgboost de validare    
training_knn_features = "training_features_knn"                 # directory ul unde vor fi salvate feature urile pt knn de antrenare
validation_knn_features = "validation_features_knn"             # directory ul unde vor fi salvate feature urile pt knn de validare

train_ratio = 80        # Define the training percentage (80% training, 20% validation)

split_files(input_dir, train_dir, val_dir, train_ratio)

xgboost_preprocessing.prepare_xgboost_features(training_dataset_path= train_dir ,  training_features_xgboost_dest=training_xgboost_features ,validation_dataset_path= val_dir ,validation_features_xgboost= validation_xgboost_features)

xboost_file = xgboost_main.train_xgboost_model(training_features_xgboost_dir_path= training_xgboost_features , validation_features_xgboost_dir_path= validation_xgboost_features)

knn_preprocessing.prepare_knn_features(xgboost_file =  xboost_file , training_features_xgboost_dir_path= training_xgboost_features  , training_features_knn_dest= training_knn_features , validation_features_xgboost_path= validation_xgboost_features, validation_features_knn_dest= validation_knn_features)

knn_main.run_knn_model(training_data_path= training_knn_features , validation_data_folder= validation_knn_features)


