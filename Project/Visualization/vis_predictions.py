import matplotlib.pyplot as plt
import numpy as np
import os
import json

def plot_sample_images_from_json(json_path, class_labels, folder="visualization", num_samples=10):
    """
    Plots and saves a specified number of sample images with their true labels from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing images as nested lists of pixel values and their class indices.
        class_labels (list): The list mapping class indices to class names.
        folder (str, optional): The directory to save the visualization. Defaults to "visualization".
        num_samples (int, optional): The number of sample images to visualize. Defaults to 10.

    Returns:
        None
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    x_test = [np.array(image_data['image_data']) for image_data in data]  # Assuming each entry contains an 'image_data' key
    true_classes = np.array([image_data['label'] for image_data in data])  # Update if your JSON uses a different key for labels
    
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    sample_images = x_test[indices]
    sample_labels = true_classes[indices]

    num_rows = int(np.ceil(num_samples / 5))
    plt.figure(figsize=(15, 3 * num_rows))
    for i, (image, label_idx) in enumerate(zip(sample_images, sample_labels)):
        plt.subplot(num_rows, 5, i + 1)
        plt.imshow(image, cmap='gray')  # Assuming grayscale for simplicity; adjust if your data is different
        plt.title(f'True: {class_labels[label_idx]}')
        plt.axis('off')

    plt.suptitle('Sample Images from JSON', fontsize=16)
    plt.tight_layout()
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(f'{folder}/sample_images_from_json.png')
    plt.close()

# Example usage
# plot_sample_images_from_json('path_to_json.json', class_labels)
