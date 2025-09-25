import tensorflow as tf
import keras
import os
import core_values as cova
gpus = tf.config.list_physical_devices('GPU')
print(gpus)

try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

except Exception as e:
    print(f"{e}")
AUTOTUNE = tf.data.AUTOTUNE
path = cova.DATA_PATH
try:
    train_ds = tf.data.Dataset.load(os.path.join(path, "train"))
    valid_ds = tf.data.Dataset.load(os.path.join(path, "valid"))

except Exception as e:
    print(f"{e}")
train = train_ds.prefetch(buffer_size = AUTOTUNE)
valid = valid_ds.prefetch(buffer_size = AUTOTUNE)
base_model = keras.applications.EfficientNetV2B3(
    include_top=False,
    weights = "imagenet",
    classes=4,
    input_shape=((cova.IMAGE_SIZE[0], cova.IMAGE_SIZE[1], 3))
)

base_model.trainable = False
model = keras.models.Sequential([
    keras.layers.Input((cova.IMAGE_SIZE[0], cova.IMAGE_SIZE[1], 3)),

    base_model,

    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(128, activation = "leaky_relu"),
    keras.layers.Dense(4, activation = "softmax")
])
early_stopping = keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    verbose = 1,
    restore_best_weights = True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.5,
    patience = 3,
    verbose = 1,
    min_lr = 0.00000001
)

terminate_nan = keras.callbacks.TerminateOnNaN()
model.compile(optimizer = "Adam",
              loss = keras.losses.SparseCategoricalCrossentropy(),
              metrics = ['accuracy'],
              steps_per_execution = 5)
model.fit(train, validation_data = valid, epochs = 256, callbacks = [reduce_lr, terminate_nan, early_stopping])
model.save('efficientnetv2b3-trained-model.keras', overwrite = True)