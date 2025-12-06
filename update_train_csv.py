# File: update_train_csv.py
import os
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
# ABSOLUTE path to the main data folder (where 'training' folder is)
DATASET_DIR = "/home/Tojo/papsmear"
TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, "training")

# RELATIVE path for the output CSV file. Saves 'train.csv' to your current working directory.
OUTPUT_CSV_PATH = "train.csv"

def update_training_csv():
    """
    Scans the training directory to create a new, updated 'train.csv' in the current
    working directory that includes all original and augmented images.
    """
    print(f"--- Starting to update CSV file at: '{os.path.abspath(OUTPUT_CSV_PATH)}' ---")

    try:
        class_names = sorted([d for d in os.listdir(TRAIN_IMAGE_DIR) if os.path.isdir(os.path.join(TRAIN_IMAGE_DIR, d))])
        if not class_names:
            print(f"❌ Error: No class subdirectories found in '{TRAIN_IMAGE_DIR}'.")
            return
        
        print(f"Found {len(class_names)} classes: {class_names}")
        class_to_int = {name: i for i, name in enumerate(class_names)}
    except FileNotFoundError:
        print(f"❌ Error: The training directory '{TRAIN_IMAGE_DIR}' was not found.")
        return

    data = []
    print("Scanning training directory to find all images...")
    
    for class_name in tqdm(class_names, desc="Processing classes"):
        class_dir = os.path.join(TRAIN_IMAGE_DIR, class_name)
        for image_name in os.listdir(class_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                relative_path = os.path.join(class_name, image_name)
                label = class_to_int[class_name]
                data.append([relative_path, label])
    
    df = pd.DataFrame(data, columns=['image_path', 'label'])
    
    print("Shuffling dataset...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) 
    
    df.to_csv(OUTPUT_CSV_PATH, index=False, header=False)
    
    print(f"✅ Successfully updated '{OUTPUT_CSV_PATH}' with {len(df)} total training samples.")

if __name__ == '__main__':
    update_training_csv()