
import pandas as pd
import os

def load_data(targetfile):
    
    if not os.path.isfile(targetfile):
        print("Invalid file path:", targetfile)
        return None
    
    data = pd.read_csv(targetfile)
    genre = targetfile.split(os.sep)[-2]
    data['genre'] = genre
    
    return data
