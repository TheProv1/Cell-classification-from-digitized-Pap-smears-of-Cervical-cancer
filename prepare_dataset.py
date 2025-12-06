import os
import glob
import numpy as np

# --- 1. Configuration ---
# All the parameters needed to define our dataset split.
# Make sure these match the expectations of your training scripts.
DATASET_DIR = "/home/Tojo/papsmear"
TRAIN_DIR = os.path.join(DATASET_DIR, "training")
SEED = 123
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.1
# The remaining 0.2 (20%) will automatically become the test set.

print("--- Starting Dataset Preparation ---")

# --- 2. Scan Files and Create Master Lists ---
try:
    # Get sorted class names to ensure consistent label encoding (0, 1, 2, 3...)
    class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    NUM_CLASSES = len(class_names)
    if NUM_CLASSES == 0:
        raise FileNotFoundError("No class subdirectories found in the training directory.")
    
    # Create a mapping from the class name (e.g., 'healthy') to an integer label (e.g., 0)
    class_to_index = {name: i for i, name in enumerate(class_names)}
    print(f"Found {NUM_CLASSES} classes: {class_names}")

    all_image_paths = []
    all_image_labels = []

    # Create a unified list of all file paths and their corresponding integer labels
    for class_name in class_names:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        # Use glob to find all files, handling different extensions like .jpg, .png, etc.
        image_paths = glob.glob(os.path.join(class_dir, '*.*'))
        all_image_paths.extend(image_paths)
        all_image_labels.extend([class_to_index[class_name]] * len(image_paths))

except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit()

# --- 3. Shuffle and Split the Data Deterministically ---
# Use a seeded random number generator for 100% reproducible shuffles.
rng = np.random.default_rng(SEED)
indices = np.arange(len(all_image_paths))
rng.shuffle(indices)

# Apply the same shuffle order to both the paths and their labels.
all_image_paths = np.array(all_image_paths)[indices]
all_image_labels = np.array(all_image_labels)[indices]

# Define the exact number of samples for each split.
num_total_samples = len(all_image_paths)
num_train = int(num_total_samples * TRAIN_SPLIT)
num_val = int(num_total_samples * VALIDATION_SPLIT)

# Manually slice the master lists into three completely separate (disjoint) sets.
train_paths = all_image_paths[:num_train]
train_labels = all_image_labels[:num_train]

val_paths = all_image_paths[num_train : num_train + num_val]
val_labels = all_image_labels[num_train : num_train + num_val]

test_paths = all_image_paths[num_train + num_val :]
test_labels = all_image_labels[num_train + num_val :]

print(f"\nTotal samples found: {num_total_samples}")
print(f"--> Training samples:   {len(train_paths)}")
print(f"--> Validation samples: {len(val_paths)}")
print(f"--> Testing samples:    {len(test_paths)}")

# --- 4. Save the Splits to Reusable CSV Files ---
# This function writes the file paths and labels to a simple text file.
def save_split(paths, labels, filename):
    with open(filename, 'w') as f:
        # Write each entry as "path/to/image.jpg,label_number"
        for path, label in zip(paths, labels):
            f.write(f"{path},{label}\n")
    print(f"✅ Saved {len(paths)} records to {filename}")

save_split(train_paths, train_labels, 'train.csv')
save_split(val_paths, val_labels, 'val.csv')
save_split(test_paths, test_labels, 'test.csv')

print("\n--- Dataset preparation complete! ---")