import json
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_misclassified_images_from_json(json_path, x_test, true_classes, class_labels, folder="visualization"):
    with open(json_path, 'r') as f:
        data = json.load(f)
    predictions = np.array(data['epoch_details'][-1]['val_predictions'])
    #predicted_classes = np.argmax(predictions, axis=1)

    misclassified_indices = np.where(predictions != true_classes)[0]
    misclassified_images = x_test[misclassified_indices]
    misclassified_true_labels = true_classes[misclassified_indices]
    misclassified_predicted_labels = predictions[misclassified_indices]

    num_misclassified = len(misclassified_images)
    num_rows = min(int(np.ceil(num_misclassified / 5)), 5)  # Limit to a maximum of 5 rows
    plt.figure(figsize=(15, 3 * num_rows))
    for i in range(min(num_misclassified, 25)):  # Limit to a maximum of 25 misclassified images
        plt.subplot(num_rows, 5, i + 1)
        #plt.imshow(misclassified_images[i])
        true_label = class_labels[misclassified_true_labels[i]]
        pred_label = class_labels[misclassified_predicted_labels[i]]
        plt.title(f'True: {true_label}, Predicted: {pred_label}')
        plt.axis('off')

    plt.suptitle('Misclassified Images', fontsize=16)
    plt.tight_layout()
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(f'{folder}/misclassified_images.png')
    plt.close()
