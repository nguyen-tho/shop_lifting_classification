import pandas as pd

import os

import config

def build_correct_path(video_name_from_csv):
    # video_name_from_csv is e.g., "Shoplifting007_x264_0"
        
    # 1. Get the base name, e.g., "Shoplifting007_x264"
    folder_name_base = "_".join(video_name_from_csv.split('_')[:-1])
        
    # 2. Create the FOLDER name, e.g., "Shoplifting007_x264.mp4"
    folder_name_full = f"{folder_name_base}.mp4" 
        
     # 3. Create the FILE name, e.g., "Shoplifting007_x264_0.mp4"
    file_name = f"{video_name_from_csv}.mp4"
        
    # 4. Join them all together
    correct_path = os.path.join(config.VIDEO_PATH, folder_name_full, file_name)
    return correct_path

def prepare_dataset():
    try:
        df = pd.read_csv(config.LABEL_CSV_PATH)
        print("Dataset loaded successfully.")
        # remove Shoplifting.csv from the path
        if "Shoplifting" in df.columns:
            df.drop("Shoplifting", axis=1, inplace=True)
            
        # Apply this new function to build the path
        df["Shoplifting001_x264_0"] = df["Shoplifting001_x264_0"].apply(build_correct_path)
    
         # Rename columns for clarity
        df = df.rename(columns={'Shoplifting001_x264_0': 'path', '0': 'target'})
    
        print(f"Successfully loaded and prepared {config.LABEL_CSV_PATH}")
        print("Example of corrected path (from CSV row 1):")
        print(df.head(1)['path'].values[0]) # Show the first corrected path

    except FileNotFoundError:
        print(f"Error: CSV file not found at '{config.LABEL_CSV_PATH}'")
        df = pd.DataFrame(columns=['path', 'target'])
        
    return df
"""
# Example usage
if __name__ == "__main__":
    dataset_df = prepare_dataset()  
    print(dataset_df.head(10))
"""