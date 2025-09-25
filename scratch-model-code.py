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
data_augmentation = keras.models.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomZoom(0.1, 0.1),
    keras.layers.RandomTranslation(0.1, 0.1),
    keras.layers.RandomRotation(0.2)
], name = "data_augmentation")
model = keras.models.Sequential([
    keras.layers.Input((cova.IMAGE_SIZE[0], cova.IMAGE_SIZE[1], 3)),

    data_augmentation,

    keras.layers.Conv2D(16, (3,3), padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(activation=keras.activations.leaky_relu),
    keras.layers.MaxPool2D((2,2)),

    keras.layers.Conv2D(32, (3,3), padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(activation=keras.activations.leaky_relu),
    keras.layers.MaxPool2D((2,2)),

    keras.layers.Conv2D(64, (3,3), padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(activation=keras.activations.leaky_relu),
    keras.layers.MaxPool2D((2,2)),

    keras.layers.Conv2D(128, (3,3), padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(activation=keras.activations.leaky_relu),
    keras.layers.MaxPool2D((2,2)),

    keras.layers.Conv2D(256, (3,3), padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(activation=keras.activations.leaky_relu),
    keras.layers.MaxPool2D((2,2)),

    keras.layers.Flatten(),

    keras.layers.LayerNormalization(),
    keras.layers.Dense(1024, activation = keras.activations.leaky_relu),
    keras.layers.Dense(64, activation = keras.activations.leaky_relu),
    keras.layers.Dense(4, activation = keras.activations.softmax)
])
model.compile(optimizer=keras.optimizers.SGD(learning_rate= 0.01),
              loss = keras.losses.SparseCategoricalCrossentropy(),
              metrics = ['accuracy'])
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
model.fit(train,  validation_data = valid, epochs = 256, callbacks=[early_stopping, reduce_lr, terminate_nan])
model.save('scratch-trained-model.keras', overwrite = True)