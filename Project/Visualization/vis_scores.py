import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_scores_from_json(json_path):
    with open(json_path) as f:
        data = json.load(f)

    all_scores = []

    for epoch in data['epochs']:
        true_labels = []
        predicted_labels = []

        for sample in epoch['samples']:
            true_labels.append(sample['label'])
            predicted_labels.append(sample['prediction'])

        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        all_scores.append([epoch['epoch_number'], precision, recall, f1])

    return all_scores

def plot_scores_across_epochs(scores, folder="visualization"):
    epochs = [score[0] for score in scores]
    precisions = [score[1] for score in scores]
    recalls = [score[2] for score in scores]
    f1_scores = [score[3] for score in scores]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, precisions, label='Precision', marker='o')
    plt.plot(epochs, recalls, label='Recall', marker='o')
    plt.plot(epochs, f1_scores, label='F1-score', marker='o')

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Score',fontsize=14)
    plt.title('Precision, Recall, and F1-score', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)

    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, 'cifar100scores_across_epochs.png'))
    plt.close()