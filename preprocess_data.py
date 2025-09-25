import tensorflow as tf
import os
import keras
import core_values as cova

AUTOTUNE = tf.data.AUTOTUNE

path = "/mnt/d/Tojo Sir - Project/"
raw_path = path + "isbi2025-ps3c-train-dataset/"
processed_path = cova.DATA_PATH
try:
    train_set = keras.utils.image_dataset_from_directory(
        raw_path,
        batch_size = 1,
        seed = cova.SEED,
        subset = "training",
        validation_split = cova.VALIDATION_SPLIT,
        shuffle = True,
        interpolation = cova.INTERPOLATION,
        label_mode = cova.LABEL_MODE,
        color_mode = cova.COLOR,
        labels = cova.LABELS,
        image_size = cova.IMAGE_SIZE 
    )

except:
    print("Error loading training set\n")
try:
    valid_set = keras.utils.image_dataset_from_directory(
        raw_path,
        batch_size = 1,
        seed = cova.SEED,
        subset = "validation",
        shuffle = False,
        validation_split = cova.VALIDATION_SPLIT,
        label_mode = cova.LABEL_MODE,
        color_mode = cova.COLOR,
        labels = cova.LABELS,
        image_size = cova.IMAGE_SIZE 
    )

except:
    print("Error loading Validation set")
def normalizeImages(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label
train_processed = train_set.map(normalizeImages, num_parallel_calls = AUTOTUNE).batch(cova.BATCH_SIZE)
valid_processed = valid_set.map(normalizeImages, num_parallel_calls = AUTOTUNE).batch(cova.BATCH_SIZE)
os.makedirs(processed_path, exist_ok = True)

tf.data.Dataset.save(train_processed, os.path.join(processed_path, 'train'))
tf.data.Dataset.save(valid_processed, os.path.join(processed_path, 'valid'))