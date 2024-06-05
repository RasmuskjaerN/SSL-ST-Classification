import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix_from_json(json_path, folder="visualization"):
    with open(json_path, 'r') as f:
        data = json.load(f)

    labels = data['epochs'][-1]['samples']
    val_true_labels = [sample['label'] for sample in labels]
    val_predictions = [sample['prediction'] for sample in labels]

    cm = confusion_matrix(val_true_labels, val_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix for CIFAR-10', fontsize=20)
    plt.xlabel('Predicted Labels', fontsize=16)
    plt.ylabel('True Labels', fontsize=16)
    plt.tight_layout()

    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(f'{folder}/metrics_and_confusion_matrix.png')
    plt.close()



def plot_confusion_matrix100_from_json(json_path, folder="visualization"):
    with open(json_path, 'r') as f:
        data = json.load(f)

    labels = data['epochs'][-1]['samples']
    val_true_labels = [sample['label'] for sample in labels]
    val_predictions = [sample['prediction'] for sample in labels]

    # Update the range to 100 for CIFAR-100
    cm = confusion_matrix(val_true_labels, val_predictions, labels=range(100))
    
    plt.figure(figsize=(15, 12))  # Increase figure size for better visualization
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=range(100), yticklabels=range(100), cbar=True)
    plt.title('Confusion Matrix for CIFAR-100', fontsize=30)
    plt.xlabel('Predicted Labels', fontsize=24)
    plt.ylabel('True Labels', fontsize=24)

    # Modify ticks to show only every 5th label
    plt.xticks(ticks=np.arange(0, 100, 5), labels=np.arange(0, 100, 5), fontsize=10)
    plt.yticks(ticks=np.arange(0, 100, 5), labels=np.arange(0, 100, 5), fontsize=10)

    plt.tight_layout()

    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(f'{folder}/metrics_and_confusion_matrix_cifar100.png')
    plt.close()
