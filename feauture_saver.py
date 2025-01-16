import os
import pandas as pd

def create_feature_file(targetDir, songName, values):
    
    print(f"Saving file for song: {songName}")

    if not os.path.exists(targetDir):
        os.makedirs(targetDir)

    fileName = f"{os.path.splitext(songName)[0]}.csv"  
    filePath = os.path.join(targetDir, fileName)
    values.to_csv(filePath, index=False)

    print(f"Feature file '{fileName}' has been {'created' if not os.path.exists(filePath) else 'updated'}")
