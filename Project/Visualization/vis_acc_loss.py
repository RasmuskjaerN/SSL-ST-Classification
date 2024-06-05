import json
import matplotlib.pyplot as plt
import os

def plot_accuracy_loss_from_json(json_path, folder="visualization"):
    with open(json_path, 'r') as f:
        data = json.load(f)

    epochs = range(1, len(data['epochs']) + 1)
    epochs_nr = len(data['epochs'])
    accuracy = [epoch['metrics']['training_accuracy'] for epoch in data['epochs']]
    val_accuracy = [epoch['metrics']['validation_accuracy'] for epoch in data['epochs']]
    loss = [epoch['metrics']['training_loss'] for epoch in data['epochs']]
    val_loss = [epoch['metrics']['validation_loss'] for epoch in data['epochs']]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, 'bo-', label='Training Acc')
    plt.plot(epochs, val_accuracy, 'ro-', label='Validation Acc')
    plt.title('Training and Validation Accuracy', fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(fontsize=16)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(fontsize=16)

    plt.suptitle(f'Accuracy / Loss, {epochs_nr} epochs', fontsize=16, y=0.98)
    plt.tight_layout()

    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(f'{folder}/Acc_Loss_Pr_Epoch.png')
    plt.close()