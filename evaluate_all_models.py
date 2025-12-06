import os
import tensorflow as tf
import keras
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import sys
from contextlib import redirect_stdout

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

os.environ['TF_CUDNN_USE_FRONTEND'] = '0'

# --- 1. Global Configuration ---
DATASET_DIR = "/home/Tojo/papsmear" 
TRAIN_DIR = os.path.join(DATASET_DIR, "training") 
TEST_TFRECORD = 'test.tfrecord'
TEST_LABELS_CSV = 'test.csv'

BATCH_SIZE = 128  
AUTOTUNE = tf.data.AUTOTUNE
TARGET_GPU = '/gpu:0'

MODEL_FILES = [
    # Lightweight & Efficient Models (Default 224x224)
    "final-mobilenet-tfrecord.keras",
    "final-nasnetmobile-tfrecord.keras",
    "final-densenet121-tfrecord.keras",
    "final-convnextsmall-tfrecord.keras",
    "final-resnet50v2-tfrecord.keras",
    "final-vgg16-tfrecord.keras",
    "final-vgg19-tfrecord.keras",

    # Models with special input sizes (299x299)
    "final-inceptionv3-tfrecord.keras",
    "final-xception-tfrecord.keras",
    
    # Model with special input size (384x384)
    "final-efficientnetv2m-tfrecord.keras",
]

# --- 2. GPU Initialization ---
print("--- Configuring GPU for Evaluation ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ GPU {gpus[0].name} (originally GPU 1) is visible and configured with memory growth.")
    except RuntimeError as e:
        print(f"❌ Error during GPU setup: {e}")
        sys.exit(1)
else:
    print("⚠️ No visible GPUs found. Check CUDA_VISIBLE_DEVICES. The script will run on the CPU.")
    TARGET_GPU = '/cpu:0'

# --- 3. Data Loading for Metadata ---
print("\n--- Loading Metadata ---")
try:
    y_true = np.loadtxt(TEST_LABELS_CSV, delimiter=',', dtype=str)[:, 1].astype(int)
    class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    NUM_CLASSES = len(class_names)
    print(f"Found {NUM_CLASSES} classes: {class_names}")
except FileNotFoundError as e:
    print(f"❌ Error: Required file not found: {e}. Please ensure '{TEST_LABELS_CSV}' and '{TRAIN_DIR}' exist.")
    sys.exit(1)

# --- 4. Dynamic Data Pipeline Creation ---
def configure_dataset(tfrecord_path, image_size, num_classes):
    """
    Configures a tf.data pipeline with a SPECIFIC image size.
    """
    if not os.path.exists(tfrecord_path):
        print(f"❌ Error: Test TFRecord file not found at '{tfrecord_path}'")
        sys.exit(1)
        
    def parse_tfrecord_fn(example):
        feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, feature_description)
        image = tf.io.decode_jpeg(example['image_raw'], channels=3)
        image = tf.image.resize(image, image_size)
        label = tf.one_hot(example['label'], depth=num_classes)
        return image, label

    ds = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=AUTOTUNE)
    ds = ds.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

# --- 5. Main Evaluation Function ---
def evaluate_model(model_path, test_dataset, true_labels, class_names_list):
    """
    Loads, compiles, evaluates, and logs performance for a single Keras model.
    """
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    log_filename = f"{model_name}_evaluation.txt"
    cm_filename = f"{model_name}_confusion_matrix.png"

    print(f"\n{'='*80}\nEvaluating model: {model_name}\n{'='*80}")

    if not os.path.exists(model_path):
        print(f"❌ SKIPPING: Model file not found at '{model_path}'")
        return

    with open(log_filename, 'w') as f, redirect_stdout(f):
        try:
            print(f"--- Loading model from: {model_path} ---")
            model = keras.models.load_model(model_path, compile=False)
            print("Model loaded successfully.")
            
            model.compile(
                loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
            print("Model compiled for evaluation.")

            print("\n\n--- Overall Test Metrics ---")
            results = model.evaluate(test_dataset, verbose=1)
            print(f"\nTest Loss: {results[0]:.4f}")
            print(f"Test Accuracy: {results[1]:.4f} ({(results[1] * 100):.2f}%)")

            print("\n--- Generating Predictions for Detailed Analysis ---")
            y_pred_logits = model.predict(test_dataset, verbose=1)
            y_pred = np.argmax(y_pred_logits, axis=1)

            print("\n--- Classification Report ---")
            print(classification_report(true_labels, y_pred, target_names=class_names_list, digits=4))

            print("\n--- Confusion Matrix ---")
            cm = confusion_matrix(true_labels, y_pred)
            print(cm)

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_list, yticklabels=class_names_list)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label'); plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(cm_filename)
            plt.close()

        except Exception as e:
            print(f"\n❌ An error occurred during the evaluation of {model_name}:")
            print(str(e))

    print(f"✅ Evaluation complete for {model_name}.")
    print(f"    📈 Detailed report saved to: '{log_filename}'")
    print(f"    📊 Confusion matrix saved to: '{cm_filename}'")

# --- 6. Main Execution Loop ---
if __name__ == "__main__":
    for model_file in MODEL_FILES:
        current_image_size = (224, 224)
        
        if "inceptionv3" in model_file or "xception" in model_file:
            current_image_size = (299, 299)
        elif "efficientnetv2m" in model_file:
            current_image_size = (384, 384)
            
        print(f"\nConfiguring data pipeline for {model_file} with image size {current_image_size}...")
        
        test_ds = configure_dataset(TEST_TFRECORD, image_size=current_image_size, num_classes=NUM_CLASSES)
        
        evaluate_model(model_file, test_ds, y_true, class_names)
        
        keras.backend.clear_session()
        print(f"🧹 Keras session cleared. GPU memory released for the next model.")
        
    print("\n\nAll models have been evaluated.")