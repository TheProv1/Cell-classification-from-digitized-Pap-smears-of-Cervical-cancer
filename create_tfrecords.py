import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

print("--- Configuring GPU Memory Growth ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Memory growth enabled for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"❌ Error setting memory growth: {e}")
else:
    print("No GPUs found, skipping GPU memory configuration.")

print("\n--- Starting TFRecord Creation (Parallelized on CPU) ---")

# --- CONFIGURATION ---
BASE_IMAGE_DIR = "/home/Tojo/papsmear/training"
AUTOTUNE = tf.data.AUTOTUNE

# --- TFRecord Helper Functions ---
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image_path_tensor, label_tensor):
    """
    Reads an image file, creates a tf.train.Example message, and serializes it.
    """
    image_raw = tf.io.read_file(image_path_tensor)
    label = tf.cast(label_tensor, tf.int64)
    
    feature = {
        'image_raw': _bytes_feature(image_raw),
        'label': _int64_feature(label),
    }
    
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(image_path, label):
    """
    Wrapper function to integrate the serialization logic into the TensorFlow graph,
    allowing it to be run in parallel by tf.data.
    """
    serialized_string = tf.py_function(
        serialize_example,
        (image_path, label),
        tf.string)
    return tf.reshape(serialized_string, ())

# --- Main Writing Function ---
def write_tfrecords_parallel(csv_path, output_path):
    """
    Reads a CSV and uses a parallel CPU-based tf.data pipeline to create the
    TFRecord file with low and stable memory usage.
    """
    print(f"Processing {csv_path} -> {output_path}")
    
    with tf.device('/cpu:0'):
        try:
            data = np.loadtxt(csv_path, delimiter=',', dtype=str)
        except FileNotFoundError:
            print(f"❌ Error: {csv_path} not found in the current directory.")
            return
            
        paths, labels = data[:, 0], data[:, 1].astype(np.int32)
        full_paths = [os.path.join(BASE_IMAGE_DIR, p) for p in paths]
        
        print(f"Building parallel pipeline for {len(paths)} records...")
        
        # 1. Build the high-performance parallel data loading pipeline
        dataset = tf.data.Dataset.from_tensor_slices((full_paths, labels))
        dataset = dataset.map(tf_serialize_example, num_parallel_calls=AUTOTUNE)
        
        # 2. Use the standard writer in an iterative "streaming" loop.
        with tf.io.TFRecordWriter(output_path) as writer:
            with tqdm(total=len(paths), unit="records", desc=f"Writing {os.path.basename(output_path)}") as pbar:
                for record in dataset:
                    writer.write(record.numpy())
                    pbar.update(1)
                
    print(f"✅ Finished writing records to {output_path}\n")

# --- Execute the Process ---
write_tfrecords_parallel('train.csv', 'train.tfrecord')
write_tfrecords_parallel('val.csv', 'val.tfrecord')
write_tfrecords_parallel('test.csv', 'test.tfrecord')

print("--- All TFRecord files created successfully! ---")