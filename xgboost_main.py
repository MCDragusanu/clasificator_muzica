import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from datetime import datetime
from feature_loader import load_data

def train_xgboost_model(training_features_xgboost_dir_path, validation_features_xgboost_dir_path):
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

                        # Map each song to its extracted data and true genre
                        song_to_frames[file_path] = song_data
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

    # Save the classifier
    model_filename = f'xgboost_classifier_{frame_accuracy:.2f}.pkl'
    joblib.dump(frame_classifier, model_filename)

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

                        filename = os.path.splitext(file)[0].split('_')[0]
                        validation_songs[filename] = features_df
                        validation_songs_true_genre[filename] = label

    validation_song_predictions = {}
    validation_mapped_frame_features = {}
    for song_path, frames in validation_songs.items():
        frame_pred = frame_classifier.predict(frames)
        frame_probabilities = frame_classifier.predict_proba(frames)
        summed_probabilities = np.sum(frame_probabilities, axis=0)
        most_likely_genre = np.argmax(summed_probabilities)
        validation_song_predictions[song_path] = most_likely_genre
        validation_mapped_frame_features[song_path] = [int(pred) for pred in frame_pred]

    true_genre_labels = list(validation_songs_true_genre.values())
    label_encoder.fit(true_genre_labels)
    encoded_true_labels = label_encoder.transform(true_genre_labels)
    validation_song_predictions_list = list(validation_song_predictions.values())

    song_accuracy = accuracy_score(encoded_true_labels, validation_song_predictions_list)
    classification_report_str = classification_report(
        encoded_true_labels,
        validation_song_predictions_list,
        target_names=label_encoder.classes_
    )
    conf_matrix = confusion_matrix(encoded_true_labels, validation_song_predictions_list)

    # Save results to a text file
    current_date = datetime.now().timestamp()
    report_filename = f'raport_clasificare_xgboost_{current_date}.txt'
    with open(report_filename, 'w') as report_file:
        report_file.write(f"Frame-Level Accuracy: {frame_accuracy:.2f}\n")
        report_file.write("\nFrame-Level Classification Report:\n")
        report_file.write(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        report_file.write("\nSong-Level Accuracy (Majority Voting):\n")
        report_file.write(f"{song_accuracy:.2f}\n")
        report_file.write("\nSong-Level Classification Report:\n")
        report_file.write(classification_report_str)
        report_file.write("\nConfusion Matrix:\n")
        report_file.write(np.array2string(conf_matrix, separator=', '))

    print(f"Results saved to {report_filename}")
    return model_filename
