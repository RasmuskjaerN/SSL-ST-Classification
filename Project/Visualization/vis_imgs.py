# import json
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# def display_images(json_path, class_labels, folder="visualization"):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
    
#     # Create a folder for saving the visualizations if it doesn't exist
#     if not os.path.exists(folder):
#         os.makedirs(folder)
    
#     for epoch in data['epochs']:
#         # Get the last 25 samples in the epoch
#         samples_with_pixels = [sample for sample in epoch['samples'] if sample['pixels'] != ""]
#         samples = samples_with_pixels[-25:]
        
#         # Calculate the number of correct and incorrect predictions
#         correct_predictions = sum(1 for sample in samples if sample['label'] == sample['prediction'])
#         incorrect_predictions = len(samples) - correct_predictions
        
#         fig, axs = plt.subplots(5, 5, figsize=(15, 15))
#         fig.suptitle(f'Epoch {epoch["epoch_number"]}: Last 25 Images', fontsize=20)
        
#         # Add a smaller subtitle below the main title
#         fig.text(0.5, 0.92, f'Correct Predictions: {correct_predictions} | Incorrect Predictions: {incorrect_predictions}', 
#                  ha='center', fontsize=14, color='gray')

#         for i, sample in enumerate(samples):
#             ax = axs[i // 5, i % 5]
#             image_data = sample['pixels']
#             true_label = sample['label']
#             predicted_label = sample['prediction']
            
#             # Convert the image data to a numpy array for plotting and remove extra dimension
#             image_array = np.squeeze(np.array(image_data))

#             # Plot the image
#             ax.imshow(image_array)
#             ax.axis('off')

#             # Set the title for true and predicted labels
#             if true_label == predicted_label:
#                 ax.set_title(f'True: {class_labels[true_label]}\nPred: {class_labels[predicted_label]}', color='green')
#             else:
#                 ax.set_title(f'True: {class_labels[true_label]}\nPred: {class_labels[predicted_label]}', color='red')

#         plt.tight_layout(rect=[0, 0.03, 1, 0.90])  # Adjust rect to make space for the subtitle
#         plt.savefig(os.path.join(folder, f'epoch_{epoch["epoch_number"]}_last_25_images.png'))
#         plt.close()


import json
import os
import numpy as np
import matplotlib.pyplot as plt

def display_images(json_path, class_labels, folder="visualization"):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create a folder for saving the visualizations if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for epoch in data['epochs']:
        # Get the last 25 samples in the epoch
        samples_with_pixels = [sample for sample in epoch['samples'] if sample['pixels'] != ""]
        samples = samples_with_pixels[-25:]
        
        # Calculate the number of correct and incorrect predictions
        correct_predictions = sum(1 for sample in samples if sample['label'] == sample['prediction'])
        incorrect_predictions = len(samples) - correct_predictions
        
        fig, axs = plt.subplots(5, 5, figsize=(15, 15))
        fig.suptitle(f'Epoch {epoch["epoch_number"]}: Last 25 Images', fontsize=26)
        
        # Add a smaller subtitle below the main title
        fig.text(0.5, 0.92, f'Correct Predictions: {correct_predictions} | Incorrect Predictions: {incorrect_predictions}', 
                 ha='center', fontsize=18)

        for i, sample in enumerate(samples):
            ax = axs[i // 5, i % 5]
            image_data = sample['pixels']
            true_label = sample['label']
            predicted_label = sample['prediction']
            
            # Convert the image data to a numpy array for plotting and remove extra dimension
            image_array = np.squeeze(np.array(image_data))

            # Plot the image
            ax.imshow(image_array)
            ax.axis('off')

            # Set the title for true and predicted labels
            if true_label == predicted_label:
                bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="green", alpha=0.7)
                ax.set_title(f'True: {class_labels[true_label]}\nPred: {class_labels[predicted_label]}', color='black', fontsize=16, bbox=bbox_props)
            else:
                bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="red", alpha=0.7)
                ax.set_title(f'True: {class_labels[true_label]}\nPred: {class_labels[predicted_label]}', color='black', fontsize=16, bbox=bbox_props)

        plt.tight_layout(rect=[0, 0.03, 1, 0.90])  # Adjust rect to make space for the subtitle
        plt.savefig(os.path.join(folder, f'epoch_{epoch["epoch_number"]}_last_25_images.png'))
        plt.close()

