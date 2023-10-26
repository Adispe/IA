import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

print('#############################')
print('####       DEBUT      #######')
print('#############################')
print('')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print('')
print('#############################')
print('')
print('Tensorflow version:')
print(tf.__version__)
print('')
print('#############################')
print('')
print('Infos GPU:')
print(tf.test.gpu_device_name())
print('#############################')
print('')

#liberation de la mémoire
tf.compat.v1.reset_default_graph()
print("Mémoire vidée OK")
print('')
print('#############################')
print('')

tf.keras.backend.clear_session()

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

csv_train_path = r"/Users/anton/Desktop/challenge_small/train_labels_GY1QjFw"
csv_test_path = r"/Users/anton/Desktop/challenge_small/test_predicted_random_rp2A5Fo.csv"
csv_list_train = r"/Users/anton/Desktop/challenge_small/train_images_Es8kvkp.csv"
csv_list_test = r"/Users/anton/Desktop/challenge_small/test_images_kkwOpBC.csv"

df_train = pd.read_csv(csv_train_path)
df_test = pd.read_csv(csv_test_path)
df_list_train = pd.read_csv(csv_list_train)
df_list_test = pd.read_csv(csv_list_test)

CLASSES = ['no_data', 'clouds', 'artificial', 'cultivated', 'broadleaf', 'coniferous', 'herbaceous', 'natural', 'snow', 'water']

image_path_train = r"/Users/anton/Desktop/challenge_small/dataset/train1/images/"
mask_path_train = r"/Users/anton/Desktop/challenge_small/dataset/train1/masks/"
image_path_test = r"/Users/anton/Desktop/challenge_small/dataset/test1/images/"

# @tf.function(jit_compile=True)
def load_data_train_batch(image_path, mask_path, batch_sample_ids):
    batch_images = []
    batch_masks = []

    for sample_id in batch_sample_ids:
        image = tf.keras.preprocessing.image.load_img(image_path + str(sample_id) + '.jpg', target_size=(256, 256))
        mask = tf.keras.preprocessing.image.load_img(mask_path + str(sample_id) + '.tif', target_size=(256, 256), color_mode='grayscale')
        batch_images.append(tf.keras.preprocessing.image.img_to_array(image))
        batch_masks.append(tf.keras.preprocessing.image.img_to_array(mask) / 255.0)

    return tf.convert_to_tensor(batch_images), tf.convert_to_tensor(batch_masks)

def load_data_train(image_path, mask_path, sample_ids, batch_size):
    num_samples = len(sample_ids)
    num_batches = int(np.ceil(num_samples / batch_size))

    def batch_generator():
        for batch_idx in tqdm(range(num_batches), desc="Loading Data Batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_sample_ids = sample_ids[start_idx:end_idx]

            batch_images, batch_masks = load_data_train_batch(image_path, mask_path, batch_sample_ids)
            yield batch_images, batch_masks

    dataset = tf.data.Dataset.from_generator(batch_generator, (tf.float32, tf.float32))
    images, masks = zip(*list(dataset))
    return tf.concat(images, axis=0), tf.concat(masks, axis=0)

# @tf.function(jit_compile=True)
def load_data_test_batch(image_path, batch_sample_ids):
    batch_images = []

    for sample_id in batch_sample_ids:
        image = tf.keras.preprocessing.image.load_img(image_path + str(sample_id) + '.jpg', target_size=(256, 256))
        batch_images.append(tf.keras.preprocessing.image.img_to_array(image))

    return tf.convert_to_tensor(batch_images)

def load_data_test(image_path, sample_ids, batch_size):
    num_samples = len(sample_ids)
    num_batches = int(np.ceil(num_samples / batch_size))

    def batch_generator():
        for batch_idx in tqdm(range(num_batches), desc="Loading Data Batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_sample_ids = sample_ids[start_idx:end_idx]

            batch_images = load_data_test_batch(image_path, batch_sample_ids)
            yield batch_images

    dataset = tf.data.Dataset.from_generator(batch_generator, tf.float32)
    images = tf.concat(list(dataset), axis=0)
    return images

sample_ids_list_train = df_list_train['sample_id'].tolist()
sample_ids_list_test = df_list_test['sample_id'].tolist()

images_train, masks_train = load_data_train(image_path_train, mask_path_train, sample_ids_list_train, 32)

images_test = load_data_test(image_path_test, sample_ids_list_test, 32 )

scaler = StandardScaler()
df_test[CLASSES] = scaler.fit_transform(df_test[CLASSES])

images_train_np = np.array(images_train)
masks_train_np = np.array(masks_train)

images_train, images_val, masks_train, masks_val = train_test_split(images_train_np, masks_train_np, test_size=0.2, random_state=42)

masks_train_one_hot = to_categorical(masks_train, num_classes=10)
masks_val_one_hot = to_categorical(masks_val, num_classes=10)

def build_unet(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv5)

    up6 = layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = layers.Conv2D(256, 2, activation='relu', padding='same')(up6)
    merge6 = layers.concatenate([conv4, up6], axis=3)
    conv6 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv6)

    up7 = layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = layers.Conv2D(128, 2, activation='relu', padding='same')(up7)
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv7)

    up8 = layers.UpSampling2D(size=(2, 2))(conv7)
    up8 = layers.Conv2D(64, 2, activation='relu', padding='same')(up8)
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv8)

    up9 = layers.UpSampling2D(size=(2, 2))(conv8)
    up9 = layers.Conv2D(32, 2, activation='relu', padding='same')(up9)
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv9)

    outputs = layers.Conv2D(10, 1, activation='softmax')(conv9)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

model = build_unet((256, 256, 3))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(images_train, masks_train_one_hot, validation_data=(images_val, masks_val_one_hot), epochs=50, batch_size=8)

model.evaluate(images_val, masks_val_one_hot, batch_size=8)

predictions = model.predict(images_test)

decoded_predictions = np.argmax(predictions, axis=-1)

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_training_history(history)