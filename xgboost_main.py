import os
import pandas as pd
from feature_loader import load_data  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier 
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np
import joblib


def train_xgboost_model(training_features_xgboost_dir_path ,   validation_features_xgboost_dir_path) :

   


    genres = os.listdir(training_features_xgboost_dir_path)

    all_frames = []
    frame_labels = []
    song_labels = []
    song_to_frames = {}
    song_true_genre = {}


    for genre in genres:
        genre_folder_path = os.path.join(training_features_xgboost_dir_path, genre)
        
        if os.path.isdir(genre_folder_path):  
            for file in os.listdir(genre_folder_path):
                if file.endswith('.csv'):  
                    file_path = os.path.join(genre_folder_path, file)
                    
                    song_data = load_data(file_path)
                    
                    if song_data is not None: 
                    
                        if 'genre' in song_data.columns:
                            song_data = song_data.drop(columns=['genre']) 
                        
                        all_frames.append(song_data)
                        frame_labels.extend([genre] * len(song_data))
                        
                        #mapeaza fieacare cantec de datele extrase
                        song_to_frames[file_path] = song_data
                        #mapeaza genul corect pt fiecare cantec
                        song_true_genre[file_path] = genre
                        song_labels.append(genre)


    frame_data = pd.concat(all_frames, ignore_index=True)


    label_encoder = LabelEncoder()
    encoded_frame_labels = label_encoder.fit_transform(frame_labels)
    encoded_song_labels = label_encoder.fit_transform(song_labels)

    X = frame_data
    y = encoded_frame_labels


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


    frame_classifier = XGBClassifier(n_estimators=100, random_state=42)
    frame_classifier.fit(X_train, y_train)



    y_pred = frame_classifier.predict(X_test)
    frame_accuracy = accuracy_score(y_test, y_pred)
    print(f'Frame-Level Accuracy: {frame_accuracy:.2f}')
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    joblib.dump(frame_classifier , f'xgboost_classifier_{frame_accuracy:.2f}.pkl')

    validation_songs = {}
    validation_songs_true_genre = {}
    labels = os.listdir(validation_features_xgboost_dir_path)


    for label in labels:
        fullPath = os.path.join(validation_features_xgboost_dir_path, label)
        
        if os.path.isdir(fullPath):
            for file in os.listdir(fullPath):
                if file.endswith('.csv'):
                    file_path = os.path.join(fullPath, file)
                    features_df = load_data(file_path)

                    if features_df is not None:
                        if 'genre' in features_df.columns:
                            features_df = features_df.drop(columns=['genre'])

                        #extrage numele fisierului fara extensie
                        filename = os.path.splitext(file)[0].split('_')[0]
                        #se mapeaza corect datele
                        validation_songs[filename] = features_df
                        validation_songs_true_genre[filename] = label

    #Se ruleaza setul de date de verificare
    validation_song_predictions = {}
    validation_mapped_frame_features = {}
    for song_path, frames in validation_songs.items():
        
        #clasifica cadrele cantecului in functie de weight-urile rezultatului clasificarii
        frame_pred = frame_classifier.predict(frames) 
        frame_probabilities = frame_classifier.predict_proba(frames)
        #acumuleaza probabilitatile claselor 
        summed_probabilities = np.sum(frame_probabilities, axis=0)
        #alege clasa cea mai frecventa
        most_likely_genre = np.argmax(summed_probabilities)
        #salveaza predictia
        validation_song_predictions[song_path] = most_likely_genre
        validation_mapped_frame_features[song_path] = [int(pred) for pred in frame_pred]
        #print(f"\nSong: {song_path}")
        #print(f"  Real Genre: {validation_songs_true_genre[song_path]}")
        #print(f"  Predicted Genre (Majority Vote): {label_encoder.inverse_transform([most_likely_genre])[0]}")
        #print(f"  Predicted Frames: {[int(pred) for pred in frame_pred]}")  # Convert to normal ints



    true_genre_labels = list(validation_songs_true_genre.values())
    label_encoder.fit(true_genre_labels)  


    encoded_true_labels = label_encoder.transform(true_genre_labels)
    validation_song_predictions_list = list(validation_song_predictions.values())


    song_accuracy = accuracy_score(encoded_true_labels, validation_song_predictions_list)
    print(f"\nSong-Level Accuracy (Majority Voting): {song_accuracy:.2f}")


    print("\nSong-Level Classification Report:")
    print(classification_report(encoded_true_labels, validation_song_predictions_list, target_names=label_encoder.classes_))


    conf_matrix = confusion_matrix(encoded_true_labels, validation_song_predictions_list)

    print("\nConfusion Matrix:")
    print(conf_matrix)


    
    return f'xgboost_classifier_{frame_accuracy:.2f}.pkl'