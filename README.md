Music Genre Classification Using XGBoost and KNN
Project Description
This project implements a music genre classification system using two machine learning models: XGBoost and K-Nearest Neighbors (KNN). The system is designed to classify songs into their respective genres based on extracted audio features. It utilizes the GTZAN dataset for training and testing.

Workflow
Download and preprocess the GTZAN dataset.
Extract audio features from song files.
Train an XGBoost model for feature transformation.
Train and test a KNN model on the transformed features.
Evaluate the performance of the models.

Features Extracted
The following features are extracted from audio windows:

Zero Crossing Rate: Measures the rate of signal-sign changes.
MFCCs (Mel-Frequency Cepstral Coefficients): Captures timbral features of the audio.
Spectral Centroid: Indicates the "center of mass" of the spectrum.
Spectral Bandwidth: Measures the range of frequencies present.
Spectral Flatness: Indicates how noise-like a signal is.
RMS Energy: Root mean square of the signal's amplitude.
Tempo: Beats per minute.

Prerequisites
Dependencies
Install the following Python libraries:
pip install pandas numpy librosa scikit-learn xgboost joblib kagglehub

