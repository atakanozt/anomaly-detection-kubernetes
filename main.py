import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

features = [
    "Flow Duration",
    "Total Fwd Packet",
    "Total Bwd packets",
    "Total Length of Fwd Packet",
    "Total Length of Bwd Packet",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Bwd IAT Total",
    "Packet Length Min",
    "Packet Length Max",
    "Packet Length Mean",
    "Packet Length Std",
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min"
]


def process_data(data):
    data = data[features]
    # Replace inf/-inf with NaN and then replace with the mean of the column
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Replace NaN values with the mean of each column (can also consider median or mode)
    data.fillna(data.mean(), inplace=True)

    # Normalize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled


def convert_to_image(data_scaled, img_dim=32):
    num_samples = data_scaled.shape[0]
    images = np.zeros((num_samples, img_dim, img_dim))

    for index in range(num_samples):
        # Flatten each sample to fit into the image dimensions
        flat_sample = np.ravel(data_scaled[index])
        # Use modulo and integer division to map the flattened data to a 2D array
        for i in range(min(flat_sample.size, img_dim * img_dim)):
            row = i // img_dim
            col = i % img_dim
            images[index, row, col] = flat_sample[i]
        # Handle case where there are fewer features than pixels
        if flat_sample.size < img_dim * img_dim:
            images[index] = np.tile(flat_sample, (img_dim * img_dim // flat_sample.size + 1))[
                            :img_dim * img_dim].reshape(img_dim, img_dim)
    return images


def show_images(images, num_images=10):
    fig, axes = plt.subplots(1, num_images, figsize=(20, 2))
    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(images[i], cmap='gray', interpolation='none')
            ax.axis('off')
        else:
            break
    plt.show()


def save_images(images, num_images=10, save_path='images/malicious/'):
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(num_images):
        fig, ax = plt.subplots()
        ax.imshow(images[i], cmap='gray', interpolation='none')
        ax.axis('off')
        # Save the figure
        fig.savefig(f"{save_path}/image_{i + 1}.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)


data = pd.read_csv("kubernetes-dataset-main/malicious.csv").sample(n=100)
data_scaled = process_data(data)
img_dim = 32
images = convert_to_image(data_scaled, img_dim)


# Visualize the images
save_images(images, num_images=100)