import os
import pandas as pd
import librosa
import numpy as np
import ntpath
from concurrent.futures import ThreadPoolExecutor, as_completed
import math



def transform_in_window(signal, sr, windowDur=1):
    window_size = math.floor(sr * windowDur)
    frames = []
    start = 0
    is_completed = False

    while not is_completed:
        fromIndex = start * window_size
        endIndex = (start + 1) * window_size
        
        if endIndex >= len(signal):  
            endIndex = len(signal)
            is_completed = True

        buffer = np.array(signal[fromIndex:endIndex])  
        frames.append(buffer) 
        start += 1

    return frames 


import librosa
import pandas as pd

def extract_features_from_window(window, sr):
    
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=window, frame_length=len(window))[0].mean()
    mel_fourier_cepstral_coeffs = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=64).mean(axis=1)
    spectral_centroid = librosa.feature.spectral_centroid(y=window, n_fft = 294 , sr=sr)[0].mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=window, n_fft = 294 ,sr=sr)[0].mean()
    spectral_flatness = librosa.feature.spectral_flatness(y=window , n_fft = 294 )[0].mean()
    rms = librosa.feature.rms(y=window)[0].mean()
    tempo = librosa.beat.tempo(y = window)[0]

    features = {
        'zero_crossing_rate': zero_crossing_rate,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_flatness': spectral_flatness,
        'rms': rms,
        'tempo':tempo
       
    }
    for i, coeff in enumerate(mel_fourier_cepstral_coeffs):
        features[f'mfcc_{i+1}'] = coeff
    
    return pd.DataFrame([features])



def extract_song_features(songPath):
    if not os.path.isfile(songPath):
        print(f"The path provided is invalid: {songPath}")
        return None
    
    if not songPath.endswith('.wav'):
        print(f"The file does not have the correct format (required .mp3): {songPath}")
        return None
    
    print(f"Processing {songPath}")

    try:
        y, sr = librosa.load(songPath)
        windows = transform_in_window(y, sr)
        features = []
        for index, window in enumerate(windows):
            window_feature = extract_features_from_window(window, sr)
            features.append(window_feature)

        all_features_df = pd.concat(features, ignore_index=True)

        return all_features_df, ntpath.basename(songPath)

    except Exception as e:
        print(f"Error processing {songPath}: {e}")
        return None



import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_features_from_genre_directory(label):
    """
    Extracts features from all .mp3 files in a directory corresponding to a specific genre.
    
    Args:
        label (str): Path to the directory containing the .wav files.
        
    Returns:
        dict: A dictionary mapping song names to their corresponding DataFrames of features.
        str: The genre label (directory name).
    """
    if not os.path.exists(label):
        print(f"No such directory found: {label}")
        return None

    print(f"Processing {label} directory...")

    # Collect all .wav files in the directory
    files = [os.path.join(label, file) for file in os.listdir(label) if file.endswith(".wav")]

    if not files:
        print(f"No valid files found in directory: {label}")
        return None, label

    song_feature_map = {}
    max_workers=4
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map file processing tasks
        future_to_file = {executor.submit(extract_song_features, file): file for file in files}
        
        # Collect results as they are completed
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                # Extract song features
                frame, song_name = future.result()
                if frame is not None:
                    # Map the DataFrame to the song name
                    song_feature_map[song_name] = frame
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    # Return the dictionary of song DataFrames and the genre label
    return song_feature_map, label
