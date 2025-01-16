from feauture_saver import create_feature_file
from feature_loader import load_data
import joblib
import os
import pandas as pd

def prepare_knn_features(xgboost_file , training_features_xgboost_dir_path, training_features_knn_dest, validation_features_xgboost_path , validation_features_knn_dest):
    
    xgboost_classifier = joblib.load(xgboost_file)

    
    labels = os.listdir(training_features_xgboost_dir_path)

    for label in labels:

        label_path = os.path.join(training_features_xgboost_dir_path , label)

        songs = os.listdir(label_path)

        for song in songs : 
            #load the frames
            song_path = os.path.join(label_path , song)

            song_data = load_data(song_path).drop(columns=['genre'])
        
        
            #classify the frames
            predicted_frames = xgboost_classifier.predict(song_data)
        
            
            frame_classes = {}

            #mapping each frame_nam
            for index in range(len(predicted_frames)):
                frame_classes[f'frame_index{index}'] = predicted_frames[index]
            
            #creating a panda_frame
            frame = pd.DataFrame([frame_classes]) 
            
            #create a new feature file
            create_feature_file(f'{training_features_knn_dest}/{label}/' , song , frame)

    
    validation_labels = os.listdir(validation_features_xgboost_path)

    for label in validation_labels:

        label_path = os.path.join(validation_features_xgboost_path , label)

        songs = os.listdir(label_path)

        for song in songs : 
            #load the frames
            song_path = os.path.join(label_path , song)

            song_data = load_data(song_path).drop(columns=['genre'])

        
            #classify the frames
            predicted_frames = xgboost_classifier.predict(song_data)
        
            
            frame_classes = {}

            #mapping each frame_nam
            for index in range(len(predicted_frames)):
                frame_classes[f'frame_index{index}'] = predicted_frames[index]
            
            #creating a panda_frame
            frame = pd.DataFrame([frame_classes]) 
            
            #create a new feature file
            create_feature_file(f'{validation_features_knn_dest}/{label}' , song , frame)