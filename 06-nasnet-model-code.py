import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CUDNN_USE_FRONTEND'] = '0'

import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from keras import Model

from keras.applications.nasnet import preprocess_input, NASNetMobile

keras.mixed_precision.set_global_policy('mixed_float16')

# --- 1. Global Configuration ---
DATASET_DIR = "/home/Tojo/papsmear"
TRAIN_DIR = os.path.join(DATASET_DIR, "training")
MODEL_SAVE_PATH = "final-nasnetmobile-tfrecord.keras"

TRAIN_TFRECORD = 'train.tfrecord'
VALID_TFRECORD = 'val.tfrecord'
TEST_TFRECORD  = 'test.tfrecord'


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 50
SEED = 123
AUTOTUNE = tf.data.AUTOTUNE
WARMUP_EPOCHS = 5
FINETUNE_LR = 4e-5

# --- 2. GPU Initialization ---
print("--- Configuring GPU for NASNetMobile Training ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ GPU {gpus[0].name} is visible and configured.")
    except RuntimeError as e:
        print(f"❌ Error during GPU setup: {e}")
else:
    print("⚠️ No GPUs found.")

# --- 3. Metadata Loading ---
try:
    train_labels = np.loadtxt('train.csv', delimiter=',', dtype=str)[:, 1].astype(int)
    test_labels = np.loadtxt('test.csv', delimiter=',', dtype=str)[:, 1].astype(int)
    class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    NUM_CLASSES = len(class_names)
except FileNotFoundError:
    print("❌ Error: train.csv or test.csv not found.")
    exit()

# --- 4. Class Weight Calculation ---
print("\n--- Calculating Class Weights ---")
total_train_samples = len(train_labels)
class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
class_weight_dict = {i: (total_train_samples / (NUM_CLASSES * count)) if count > 0 else 0 for i, count in enumerate(class_counts)}
for i, w in class_weight_dict.items(): print(f"Class '{class_names[i]}' weight: {w:.2f}")

# --- 5. High-Performance TFRecord Pipeline ---
def parse_tfrecord_fn(example):
    feature_description = { 'image_raw': tf.io.FixedLenFeature([], tf.string), 'label': tf.io.FixedLenFeature([], tf.int64) }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example['image_raw'], channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    label = tf.one_hot(example['label'], depth=NUM_CLASSES)
    return image, label

def configure_dataset(tfrecord_path, shuffle=False):
    ds = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=AUTOTUNE)
    ds = ds.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=2048, seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    return ds

print("\n--- Building High-Performance TFRecord Data Pipelines ---")
train_ds = configure_dataset(TRAIN_TFRECORD, shuffle=True)
valid_ds = configure_dataset(VALID_TFRECORD)
test_ds  = configure_dataset(TEST_TFRECORD)

# --- 6. Model Definition ---
with tf.device('/gpu:0'):
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.2),
    ], name="data_augmentation")

    base_model = NASNetMobile(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))
    base_model.trainable = False

    inputs = keras.layers.Input(shape=(*IMAGE_SIZE, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, dtype='float32')(x)
    model = Model(inputs, outputs)

    loss_function = keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer_warmup = keras.optimizers.AdamW(learning_rate=1e-4)
    model.compile(optimizer=optimizer_warmup, loss=loss_function, metrics=['accuracy'])
    model.summary()

# --- 7. Callbacks ---
callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor="val_loss", save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
]

# --- 8. Two-Stage Training ---
print(f"\n--- WARM-UP Training for {WARMUP_EPOCHS} epochs ---")
history_warmup = model.fit(train_ds, epochs=WARMUP_EPOCHS, validation_data=valid_ds, callbacks=callbacks, class_weight=class_weight_dict, verbose=1)

print("\n--- FINE-TUNING ---")
base_model.trainable = True
model.load_weights(MODEL_SAVE_PATH)
optimizer_finetune = keras.optimizers.AdamW(learning_rate=FINETUNE_LR)
model.compile(optimizer=optimizer_finetune, loss=loss_function, metrics=['accuracy'])
history_finetune = model.fit(train_ds, epochs=EPOCHS, initial_epoch=history_warmup.epoch[-1] + 1, validation_data=valid_ds, callbacks=callbacks, class_weight=class_weight_dict, verbose=1)

# --- 9. Final Evaluation ---
print("\n✅ Training complete.")
best_model = keras.models.load_model(MODEL_SAVE_PATH, compile=False)
results = best_model.evaluate(test_ds, verbose=0)
print(f"\nTest Loss: {results[0]:.4f} | Test Accuracy: {results[1]:.4f}")

y_pred_logits = best_model.predict(test_ds)
y_pred = np.argmax(y_pred_logits, axis=1)
y_true = test_labels

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - NASNetMobile')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.savefig('nasnetmobile_confusion_matrix.png')
print("\n✅ Confusion matrix saved to nasnetmobile_confusion_matrix.png")