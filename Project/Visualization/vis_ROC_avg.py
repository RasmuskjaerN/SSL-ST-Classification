import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

def plot_roc_avg(json_path, n_classes=100, folder="visualization"):
    # Load results from JSON
    with open(json_path, 'r') as file:
        data = json.load(file)
        # Extract the val_confidences and val_true_labels from the last epoch
        y_score = np.array([sample['confidences'] for sample in data['epochs'][-1]['samples']])
        y_true = np.array([sample['label'] for sample in data['epochs'][-1]['samples']])

    # Binarize the labels
    y_test = label_binarize(y_true, classes=range(n_classes))

    # Compute micro-average ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    
    # Youden's J statistic to find the optimal threshold
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]

    # Plot micro-average ROC curve
    plt.figure(figsize=(8, 8))
    lw = 2
    plt.plot(fpr, tpr, color='red', lw=lw)

    # Draw a line from the Youden's J statistic point to the bottom of the plot
    plt.plot([fpr[optimal_idx], fpr[optimal_idx]], [0, tpr[optimal_idx]], 'k-', lw=1)

    # Add a black dot for the Youden's J statistic on the curve
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ko')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-average ROC Curve across All Classes')

    # Create custom legend handles
    from matplotlib.lines import Line2D
    legend_lines = [Line2D([0], [0], color='red', lw=lw),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10)]

    # Create custom legend labels
    legend_labels = [f'Micro-average ROC curve (area = {roc_auc:0.2f})',
                     f'Youden\'s J statistic (optimal threshold): {optimal_threshold:.2f}']

    # Add legend to the plot
    plt.legend(legend_lines, legend_labels, loc="lower right")

    # Create a zoomed-in view of the top-left corner
    ax_inset = inset_axes(plt.gca(), width=1.2, height=1.2, loc="center right")
    ax_inset.plot(fpr, tpr, color='red', lw=lw)
    ax_inset.plot(fpr[optimal_idx], tpr[optimal_idx], 'ko')
    ax_inset.set_xlim(0, 0.2)
    ax_inset.set_ylim(0.8, 1)
    ax_inset.set_title('Zoom at top left')

    # Ensure the directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(f'{folder}/ROC_AVG.png')
    plt.close()



# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# from sklearn.preprocessing import label_binarize
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import os

# def plot_roc_curve_2_from_json(json_path, n_classes=10, folder="visualization"):
#     # Load results from JSON
#     with open(json_path, 'r') as file:
#         data = json.load(file)
#         y_score = np.array(data['epoch_details'][-1]['val_confidences'])

#     # Binarize the labels if not already done
#     y_test = label_binarize(data['epoch_details'][-1]['val_true_labels'], classes=range(n_classes))

#     # Compute micro-average ROC curve and ROC area
#     fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_score.ravel())
#     roc_auc = auc(fpr, tpr)
    
#     # Youden's J statistic to find the optimal threshold
#     youden_j = tpr - fpr
#     optimal_idx = np.argmax(youden_j)
#     optimal_threshold = thresholds[optimal_idx]

#     # Plot micro-average ROC curve
#     plt.figure(figsize=(8, 8))
#     lw = 2
#     plt.plot(fpr, tpr, color='red', lw=lw)

#     # Draw a line from the Youden's J statistic point to the bottom of the plot
#     plt.plot([fpr[optimal_idx], fpr[optimal_idx]], [0, tpr[optimal_idx]], 'k-', lw=1)

#     # Add a black dot for the Youden's J statistic on the curve
#     plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ko')

#     plt.plot([0, 1], [0, 1], 'k--', lw=1)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.0])
#     plt.xlabel('1-Specificity')
#     plt.ylabel('Sensitivity')
#     plt.title('Micro-average ROC Curve across All Classes')

#     # Create custom legend handles
#     from matplotlib.lines import Line2D
#     legend_lines = [Line2D([0], [0], color='red', lw=lw),
#                     Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10)]

#     # Create custom legend labels
#     legend_labels = [f'Micro-average ROC curve (area = {roc_auc:0.2f})',
#                      f'Youden\'s J statistic (optimal threshold): {optimal_threshold:.2f}']

#     # Add legend to the plot
#     plt.legend(legend_lines, legend_labels, loc="lower right")

#     # Create a zoomed-in view of the top-left corner
#     ax_inset = inset_axes(plt.gca(), width=1.2, height=1.2, loc="center right")
#     ax_inset.plot(fpr, tpr, color='red', lw=lw)
#     ax_inset.plot(fpr[optimal_idx], tpr[optimal_idx], 'ko')
#     ax_inset.set_xlim(0, 0.2)
#     ax_inset.set_ylim(0.8, 1)
#     ax_inset.set_title('Zoom at top left')

#     # Ensure the directory exists
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     plt.savefig(f'{folder}/ROC_Single_Chart.png')
#     plt.close()
