import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt


def get_siamese_model(input_shape):
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Define a Convolutional Neural Network architecture
    model = tf.keras.Sequential([
        Conv2D(64, (5, 5), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(4096, activation='sigmoid')
    ])

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # Return the model
    return siamese_net


# Assuming images are 32x32 grayscale
model = get_siamese_model((32, 32, 1))
model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])


def load_images_from_directory(directory, target_size=(32, 32), max_images=None):
    images = []
    filenames = os.listdir(directory)
    loaded_images = 0
    for filename in filenames:
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            img = load_img(img_path, target_size=target_size, color_mode='grayscale')
            img_array = img_to_array(img)
            images.append(img_array)
            loaded_images += 1
            if max_images is not None and loaded_images >= max_images:
                break
    images = np.array(images)
    return images


# Paths to your image directories. IT IS IMPORTANT THAT HOW MANY IMAGES IN YOUR THIS FOLDER. THIS LIMITS THE FEW-SHOT ALGORITHM
# WE USED 3-10 benign images and 1-5 malicious images at most. It is recommended to use this number of images.
benign_dir = 'images/benign'
malicious_dir = 'images/malicious'

benign_images = load_images_from_directory(benign_dir, target_size=(32, 32), max_images=10)
malicious_images = load_images_from_directory(malicious_dir, target_size=(32, 32), max_images=1)

print(f"Loaded benign images shape: {benign_images.shape}")
print(f"Loaded malicious images shape: {malicious_images.shape}")


def create_pairs(benign_images_pairing, malicious_images_pairing):
    """
    IMPORTANT: It is important that how many images in your saved_images folder. For few-shot algorithm we used 3 benign and 1 malicious images.
                Or used at most 10 benign and 3 malicious images.


    :param benign_images_pairing:
    :param malicious_images_pairing:
    :return: pairs and labels
    """
    pairs = []
    labels = []

    # Benign-Benign pairs
    for i in range(len(benign_images_pairing) - 1):
        for j in range(i + 1, len(benign_images_pairing) - 1):
            pairs.append([benign_images_pairing[i], benign_images_pairing[j]])
            labels.append(1)

    # Benign-Malicious pairs
    for benign_image in benign_images_pairing:
        for malicious_image in malicious_images_pairing:
            pairs.append([malicious_image, benign_image])
            labels.append(0)

    pairs = np.array(pairs)
    labels = np.array(labels)
    return pairs, labels


pairs, labels = create_pairs(benign_images, malicious_images)

# Check shapes
left_images = np.array([pair[0] for pair in pairs])
right_images = np.array([pair[1] for pair in pairs])

print(f"Left images shape after pairing: {left_images.shape}")
print(f"Right images shape after pairing: {right_images.shape}")

early_stopping = EarlyStopping(monitor='accuracy', patience=3, restore_best_weights=True)

print(f"Left images: {len(left_images)}")
print(f"Right images: {len(right_images)}")

print(f"labels: {labels}")
print(f"len labels: {len(labels)}")

# Train the model
history = model.fit(
    [left_images, right_images],  # input pairs
    labels,  # corresponding labels
    batch_size=5,  # batch size for training
    epochs=20,  # number of epochs to train
    validation_split=0.2,  # percentage of data to use for validation
    callbacks=[early_stopping]  # early stopping
)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
