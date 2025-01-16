import os
import pandas as pd
from feature_loader import load_data
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

def run_knn_model(training_data_path, validation_data_folder):
    labels = os.listdir(training_data_path)

    training_set = []
    song_to_true_label = {}
    song_to_frames = {}
    training_labels = []

    for label in labels:
        song_data = []
        path = os.path.join(training_data_path, label)
        labeled_song_files = os.listdir(path)

        for labeled_song_file in labeled_song_files:
            song_path = os.path.join(path, labeled_song_file)
            song_frames = load_data(song_path)

            if 'genre' in song_frames.columns:
                song_frames = song_frames.drop(columns=["genre"])

            print(f"Finished loading: {labeled_song_file}")
            song_data.append(song_frames)
            song_to_true_label[labeled_song_file] = label
            song_to_frames[labeled_song_file] = song_frames

        training_set.append(song_data)
        training_labels.extend([label] * len(song_data))
        print(f" --------  Finished loading label: {label}  ---------")

    label_encoder = LabelEncoder()
    encoded_training_labels = label_encoder.fit_transform(training_labels)

    tmp = [frame for genre_data in training_set for frame in genre_data]
    X = [frame.values.flatten() if isinstance(frame, pd.DataFrame) else np.array(frame).flatten() for frame in tmp]
    y = encoded_training_labels

    X_filtered = []
    y_filtered = []
    for i, frame in enumerate(X):
        if np.shape(frame)[0] == 31:  
            X_filtered.append(frame)
            y_filtered.append(y[i])
        else:
            print(f"Removed frame {i} due to wrong dimension!")

    X = np.array(X_filtered)
    y = np.array(y_filtered)

    print(f"Lungime intrari dupa filtrare: {len(X)}  Lungime iesiri dupa filtrare: {len(y)}")
    print(f"Final shape of X: {X.shape}")
    print(f"Final shape of y: {y.shape}")

    knn = KNeighborsClassifier(n_neighbors=21, metric='hamming')
    knn.fit(X, y)

    test_labels = os.listdir(validation_data_folder)
    test_song_to_true_label = {}
    X_test = []
    y_test = []
    test_song_names = []

    for label in test_labels:
        path = os.path.join(validation_data_folder, label)
        test_files = os.listdir(path)

        for test_file in test_files:
            song_path = os.path.join(path, test_file)
            test_frames = load_data(song_path)

            if 'genre' in test_frames.columns:
                test_frames = test_frames.drop(columns=["genre"])

            flattened_frame = test_frames.values.flatten() if isinstance(test_frames, pd.DataFrame) else np.array(test_frames).flatten()
            if flattened_frame.shape[0] == 31:  
                X_test.append(flattened_frame)
                test_song_to_true_label[test_file] = label
                y_test.append(label)
                test_song_names.append(test_file)

    k = 2  
    test_encoded_labels = label_encoder.transform(y_test)
    test_probabilities = knn.predict_proba(X_test)
    top_k_predictions = []

    for probs in test_probabilities:
        top_k_indices = np.argsort(probs)[-k:][::-1]
        top_k_labels = label_encoder.inverse_transform(top_k_indices)
        top_k_predictions.append(top_k_labels)

    correctly_classified_top_k = 0
    for true_label, top_labels in zip(test_encoded_labels, top_k_predictions):
        true_label_decoded = label_encoder.inverse_transform([true_label])[0]
        if true_label_decoded in top_labels:
            correctly_classified_top_k += 1

    test_accuracy_top_k = correctly_classified_top_k / len(test_song_names)
    y_pred = knn.predict(X_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred_decoded)
    class_report = classification_report(y_test, y_pred_decoded)

    # Save results to a text file
    current_date = datetime.now().strftime('%Y-%m-%d %hh-%mm')
    report_filename = f'raport_clasificare_knn_{current_date}.txt'
    with open(report_filename, 'w') as report_file:
        report_file.write(f"Top-{k} Accuracy: {test_accuracy_top_k:.2f}\n")
        report_file.write("\nConfusion Matrix:\n")
        report_file.write(np.array2string(conf_matrix, separator=', '))
        report_file.write("\n\nClassification Report:\n")
        report_file.write(class_report)

    print(f"Results saved to {report_filename}")