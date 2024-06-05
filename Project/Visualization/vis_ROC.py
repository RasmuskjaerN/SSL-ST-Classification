import json
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_roc(json_path, n_classes, class_labels, folder="visualization"):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract the true labels and confidences from the last epoch
    last_epoch = data['epochs'][-1]
    y_true = np.array([sample['label'] for sample in last_epoch['samples']])
    y_score = np.array([sample['confidences'] for sample in last_epoch['samples']])

    # Binarize the true labels for multi-class ROC computation
    y_test = label_binarize(y_true, classes=range(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_map = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        color_map[i] = colors[i % len(colors)]  # Assign colors cyclically and store them

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    lw = 2

    # Plot the micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    # Plot each class's ROC curve
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=color_map[i], lw=lw,
                 label=f'class {class_labels[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")

    # Inset for zoomed-in ROC curve
    axins = inset_axes(plt.gca(), width="30%", height="30%", loc='upper right', borderpad=4)
    axins.set_xlim(-0.01, 0.2)  # Zoomed x-axis
    axins.set_ylim(0.8, 1.01)  # Zoomed y-axis
    for i in range(n_classes):
        axins.plot(fpr[i], tpr[i], color=color_map[i], lw=2)
    axins.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=2)
    
    axins.tick_params(axis='both', which='major', labelsize=10)
    axins.set_aspect('equal', 'box')

    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(f'{folder}/ROC_Curves.png')
    plt.close()


# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from sklearn.metrics import roc_curve, auc
# from sklearn.preprocessing import label_binarize
# from itertools import cycle
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# def plot_roc_curve_from_json(json_path, n_classes, class_labels, folder="visualization"):
#     with open(json_path, 'r') as f:
#         data = json.load(f)

#     # Predictions from JSON and true classes are assumed to be provided
#     y_score = np.array(data['epoch_details'][-1]['val_confidences'])
#     y_test = label_binarize(data['epoch_details'][-1]['val_true_labels'], classes=range(n_classes))

#     # Compute ROC curve and ROC area for each class
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
#     color_map = {}

#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#         color_map[i] = colors[i % len(colors)]  # Assign colors cyclically and store them

#     # Compute micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#     # Plot ROC curve
#     plt.figure(figsize=(10, 8))
#     lw = 2

#     # Plot the micro-average ROC curve
#     plt.plot(fpr["micro"], tpr["micro"],
#              label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
#              color='deeppink', linestyle=':', linewidth=4)

#     # Plot each class's ROC curve
#     for i in range(n_classes):
#         plt.plot(fpr[i], tpr[i], color=color_map[i], lw=lw,
#                  label=f'class {class_labels[i]} (area = {roc_auc[i]:0.2f})')

#     plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curves')
#     plt.legend(loc="lower right")

#     # Inset for zoomed-in ROC curve
#     axins = inset_axes(plt.gca(), width="30%", height="30%", loc='upper right', borderpad=4)
#     axins.set_xlim(-0.01, 0.2)  # Zoomed x-axis
#     axins.set_ylim(0.8, 1.01)  # Zoomed y-axis
#     for i in range(n_classes):
#         axins.plot(fpr[i], tpr[i], color=color_map[i], lw=2)
#     axins.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=2)
    
#     axins.tick_params(axis='both', which='major', labelsize=10)
#     axins.set_aspect('equal', 'box')

#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     plt.savefig(f'{folder}/ROC_Curves.png')
#     plt.close()
