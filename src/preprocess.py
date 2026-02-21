import os
import cv2
import random
import shutil
# --- 3. Frame Extraction Function ---
def extract_frames(video_path, target, base_output_dir, start_index):
    frame_count = 0
    
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found {video_path}. Skipping.")
        return

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}. Skipping.")
        return

    # Define "normal" or "abnormal" sub-directory
    if target == 0:
        frame_dir = os.path.join(base_output_dir, "normal")
    else:
        frame_dir = os.path.join(base_output_dir, "abnormal")
    
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
            
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imwrite(os.path.join(frame_dir, f"frame_{start_index + frame_count}.jpg"), frame)
        frame_count += 1
    
    cap.release()
    
def process_videos(df, output_dir):
    # --- 4. Run the Frame Extraction ---
    if not df.empty:
        if not os.path.exists(output_dir):
             os.makedirs(output_dir)

        i = 0
        print("\nStarting frame extraction...")
        for index, row in df.iterrows():
            video_path = row['path'].strip()
            target = row['target']
        
        extract_frames(video_path, target, output_dir, i)
        # Increment by a large number to ensure unique frame names
        i += 1000000 

        print("Frame extraction completed.")
    else:
        print("DataFrame is empty. Cannot extract frames.")
    
    
def split_data(source_folder, test_folder, split_ratio=0.2):
    """Randomly moves a percentage of files to a test directory."""
    
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
        
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' not found. Cannot split.")
        return

    files = os.listdir(source_folder)
    
    if not files:
        print(f"Warning: No files found in '{source_folder}'. Nothing to split.")
        return
        
    num_files_to_move = int(split_ratio * len(files))
    
    if num_files_to_move == 0 and len(files) > 0:
        num_files_to_move = 1

    files_to_move = random.sample(files, num_files_to_move)

    for file_name in files_to_move:
        src = os.path.join(source_folder, file_name)
        dst = os.path.join(test_folder, file_name)
        shutil.move(src, dst)
    
    print(f"Moved {len(files_to_move)} images from '{source_folder}' to '{test_folder}'.")
