from functions import *
import os
import numpy as np
import numpy as np
import pathlib
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf_version = int(tf.__version__.split(".", maxsplit=1)[0])


class_names = []
def make_dataset(data_dir, include_label=False):
    global class_names
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print('image count : ',image_count)
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
    print('classes num : ',len(class_names))
    print(class_names)
    val_size = int(image_count * 0.1)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size) 
    print('train image count : ',tf.data.experimental.cardinality(train_ds).numpy())
    print('val image count : ',tf.data.experimental.cardinality(val_ds).numpy())
    if include_label:
        train_ds = train_ds.map(process_pair_augment_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.map(process_pair_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        train_ds = train_ds.map(process_augment_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for image, label in train_ds.take(1):
        print("Image shape: ", image[0].numpy().shape)
        print("Label: ", label.numpy())
    train_ds = configure_for_performance(train_ds,64)
    val_ds = configure_for_performance(val_ds,64)
    return train_ds, val_ds

def make_dataset_noaug(data_dir, include_label=False):
    global class_names
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print('image count : ',image_count)
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
    print('classes num : ',len(class_names))
    print(class_names)
    val_size = int(image_count * 0.1)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size) 
    print('train image count : ',tf.data.experimental.cardinality(train_ds).numpy())
    print('val image count : ',tf.data.experimental.cardinality(val_ds).numpy())
    if include_label:
        train_ds = train_ds.map(process_pair_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.map(process_pair_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for image, label in train_ds.take(1):
        print("Image shape: ", image[0].numpy().shape)
        print("Label: ", label.numpy())
    train_ds = configure_for_performance(train_ds,64)
    val_ds = configure_for_performance(val_ds,64)
    return train_ds, val_ds

def get_label(file_path):
    global class_names
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)

def decode_augment_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    #img = tf.image.resize(img, [112, 112])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_saturation(img, 0.6, 1.4)
    img = tf.image.random_brightness(img, 0.4)
    img = img / 255
    return img

def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    img = img / 255
    return img

def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    #return (img, label), label
    return img, label

def process_augment_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_augment_img(img)
    #return (img, label), label
    return img, label

def process_pair_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return (img, label), label

def process_pair_augment_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_augment_img(img)
    return (img, label), label

def configure_for_performance(ds, batch_size):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds