import json
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_top_confidences(json_path, class_labels, folder="visualization"):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create a folder for saving the visualizations if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for epoch in data['epochs']:
        # Get the last 25 samples in the epoch
        #samples_with_pixels = [sample for sample in epoch['samples'] if sample['pixels'] != ""]
        #samples = samples_with_pixels[-25:]
        samples_all = epoch['samples']
        samples = samples_all[-25:]
        
        
        fig, axs = plt.subplots(5, 5, figsize=(20, 20))
        fig.suptitle(f'Epoch {epoch["epoch_number"]}: Top 5 Confidences for Last 25 Images', fontsize=32)
        
        for i, sample in enumerate(samples):
            ax = axs[i // 5, i % 5]
            image_id = sample['image_id']
            true_label = sample['label']
            confidences = sample['confidences']
            
            # Get the top 5 confidences and their indices
            top_5_confidences = sorted(enumerate(confidences), key=lambda x: x[1], reverse=True)[:5]
            top_5_indices = [index for index, confidence in top_5_confidences]
            top_5_values = [confidence for index, confidence in top_5_confidences]
            top_5_labels = [class_labels[index] for index in top_5_indices]
            
            # Determine colors for the bars
            colors = ['green' if index == true_label else 'red' for index in top_5_indices]
            
            # Plot the top 5 confidences as a bar chart
            ax.bar(top_5_labels, top_5_values, color=colors)
            ax.set_ylim(0, 1)
            ax.set_title(f'Image {i + 1} (True: {class_labels[true_label]})', fontsize=20)
            ax.set_xlabel('Class', fontsize=18)
            ax.set_ylabel('Confidence', fontsize=18)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to make space for the title
        plt.savefig(os.path.join(folder, f'cifar100epoch_{epoch["epoch_number"]}_top_5_confidences.png'))
        plt.close()