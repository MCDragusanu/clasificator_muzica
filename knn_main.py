import os
import pandas as pd
from feature_loader import load_data
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

def run_knn_model(training_data_path , validation_data_folder) : 
    

    
    labels = os.listdir(training_data_path)

    training_set = []

    song_to_true_label = {}

    song_to_frames = {}

    training_labels = []

    for label in labels : 

        #pentru fiecare genre aici se for stoca cantecele individuale
        song_data = []

        path = os.path.join(training_data_path ,  label)

        labeled_song_files = os.listdir(path)

        for labeled_song_file in labeled_song_files:

            song_path = os.path.join(path , labeled_song_file)

            song_frames = load_data(song_path)

           
            if 'genre' in song_frames.columns:
                song_frames = song_frames.drop(columns=["genre"])


            print(f"Finished loading : {labeled_song_file}")

            
            #adaug cantecul curent
            song_data.append(song_frames)

            #mappez labelul pentru fiecare cantec
            song_to_true_label[labeled_song_file] = label

            #mapez frame urile pentru fiecare cantec
            song_to_frames[labeled_song_file] = song_frames

        #extind setul de date cu toate cantecele din acea clasa
        training_set.append(song_data) 

        #mapez fiecare feature vector din fiecare folder de labelul corect
        training_labels.extend([label] * len(song_data))
        
        print(f" --------  Finished loading label : {label}  ---------")


    label_encoder = LabelEncoder()
    encoded_training_labels = label_encoder.fit_transform(training_labels)

    tmp = [frame for genre_data in training_set for frame in genre_data]
    X = [frame.values.flatten() if isinstance(frame, pd.DataFrame) else np.array(frame).flatten() for frame in tmp]
    y = encoded_training_labels

    
    X_filtered = []
    y_filtered = []

    for i, frame in enumerate(X):
        if np.shape(frame)[0] == 31:  # Unele cantece nu au dimensiunea exacta
            X_filtered.append(frame)
            y_filtered.append(y[i])
        else:
            print(f"Removed frame {i} due to wrong dimension!")

  
    X = np.array(X_filtered)
    y = np.array(y_filtered)

    print(f"Lungime intrari dupa filtrare : {len(X)}  Lungime iesiri dupa filtrare : {len(y)}")
    print(f"Final shape of X: {X.shape}")
    print(f"Final shape of y: {y.shape}")

    knn = KNeighborsClassifier(n_neighbors=21, metric='hamming',) 
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
            if flattened_frame.shape[0] == 31:  # Ensure correct shape
                X_test.append(flattened_frame)
                test_song_to_true_label[test_file] = label
                y_test.append(label)
                test_song_names.append(test_file)

    k = 2  #Top k cele mai bune predictii. Alege primele k clasele cu prob cea mai mare

    test_encoded_labels = label_encoder.transform(y_test)  # Encode true labels
    

    test_probabilities = knn.predict_proba(X_test)  
    top_k_predictions = []

    for probs in test_probabilities:
        # Get indices of the k largest probabilities
        top_k_indices = np.argsort(probs)[-k:][::-1]
        
        # Get the corresponding labels for those indices
        top_k_labels = label_encoder.inverse_transform(top_k_indices)
        
        # Store the top k predicted labels
        top_k_predictions.append(top_k_labels)

    # Evaluate based on top k predictions
    correctly_classified_top_k = 0
    for true_label, top_labels in zip(test_encoded_labels, top_k_predictions):
        true_label_decoded = label_encoder.inverse_transform([true_label])[0]
        if true_label_decoded in top_labels:
            correctly_classified_top_k += 1



    # Calculate accuracy for top k predictions
    test_accuracy_top_k = correctly_classified_top_k / len(test_song_names)
    # Calculate confusion matrix for top k predictions
    test_conf_matrix_top_k = np.zeros((len(label_encoder.classes_), len(label_encoder.classes_)))

    for true_label, top_labels in zip(test_encoded_labels, top_k_predictions):
        true_label_decoded = label_encoder.inverse_transform([true_label])[0]
        for pred_label in top_labels:
            true_idx = label_encoder.transform([true_label_decoded])[0]
            pred_idx = label_encoder.transform([pred_label])[0]
            test_conf_matrix_top_k[true_idx][pred_idx] += 1


    # Convert the top predicted labels to their corresponding encoded numeric values
    top_k_predictions_encoded = [
        label_encoder.transform([label])[0]  # Convert top label to encoded numeric value
        for top_labels in top_k_predictions
        for label in top_labels[:1]  # We can consider just the first of the top k for classification
    ]


    # Predict
    y_pred = knn.predict(X_test)

    # Decode the predicted labels (numeric to string)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    # Calculate the confusion matrix using decoded string labels
    conf_matrix = confusion_matrix(y_test, y_pred_decoded)

    # Calculate the classification report using decoded string labels
    class_report = classification_report(y_test, y_pred_decoded)

    print("\n\n------------ Final Results ----------------")
    print(f"Accuracy: {test_accuracy_top_k:.2f}\nConfusion Matrix:\n{conf_matrix}\nClassification Report:\n{class_report}")