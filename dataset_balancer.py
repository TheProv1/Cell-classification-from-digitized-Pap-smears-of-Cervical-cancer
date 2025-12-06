# File: dataset_balancer.py
import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from collections import Counter
from tqdm import tqdm

# --- Configuration ---
DATASET_DIR = "/home/Tojo/papsmear"
TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, "training")

CSV_PATH = 'train.csv'

# --- Augmentation Pipeline ---
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30, p=0.7),
    A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03), # Simulates tissue warping

    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),

    A.GaussNoise(var_limit=(5.0, 20.0), p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
])

def balance_dataset():
    """
    Reads 'train.csv' from the current directory, identifies class imbalances,
    and generates augmented images for minority classes to balance the dataset.
    """
    print("--- Starting Dataset Balancing ---")
    print(f"Reading initial training data from: '{os.path.abspath(CSV_PATH)}'")

    try:
        train_df = pd.read_csv(CSV_PATH, header=None, names=['image_path', 'label'])
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{CSV_PATH}' in your current directory. Please run 'prepare_dataset.py' here first.")
        return

    CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_IMAGE_DIR) if os.path.isdir(os.path.join(TRAIN_IMAGE_DIR, d))])
    class_counts = Counter(train_df['label'])
    if not class_counts:
        print("❌ Error: The CSV file seems to be empty or in an incorrect format.")
        return

    max_count = max(class_counts.values())
    print(f"Original class distribution: {class_counts}")
    print(f"Target number of images per class: {max_count}")

    for class_label, count in class_counts.items():
        if count < max_count:
            num_to_generate = max_count - count
            print(f"\nAugmenting class '{CLASS_NAMES[class_label]}'. Need to generate {num_to_generate} images.")

            class_images_df = train_df[train_df['label'] == class_label]
            image_paths = class_images_df['image_path'].tolist()

            generated_count = 0
            with tqdm(total=num_to_generate, desc=f"Generating for class {CLASS_NAMES[class_label]}") as pbar:
                while generated_count < num_to_generate:
                    random_image_path = np.random.choice(image_paths)
                    full_image_path = os.path.join(TRAIN_IMAGE_DIR, random_image_path)
                    
                    try:
                        image = cv2.imread(full_image_path)
                        if image is None: continue
                        
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        augmented = augmentation_pipeline(image=image)
                        augmented_image = augmented['image']

                        original_filename = os.path.basename(random_image_path)
                        filename, ext = os.path.splitext(original_filename)
                        new_filename = f"{filename}_aug_{generated_count}{ext}"
                        
                        class_dir = os.path.dirname(random_image_path)
                        save_path = os.path.join(TRAIN_IMAGE_DIR, class_dir, new_filename)
                        
                        save_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_path, save_image)
                        
                        generated_count += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"An error occurred while processing {random_image_path}: {e}")

    print("\n--- Dataset Balancing Complete ---")
    print("Next step: Run 'update_train_csv.py' to update your CSV file.")

if __name__ == '__main__':
    balance_dataset()