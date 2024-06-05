import json
import numpy as np
import os

#----- load visualizations here -----#
from vis_acc_loss import plot_accuracy_loss_from_json
from vis_ROC_avg import plot_roc_avg
from vis_confusion_matrix import plot_confusion_matrix_from_json, plot_confusion_matrix100_from_json
from vis_imgs import display_images
from vis_ROC import plot_roc
from vis_confidence_bars import plot_top_confidences
from vis_scores import calculate_scores_from_json, plot_scores_across_epochs

#----- Change between cifar 10 or 100 here -----#
cifar_version = 10

cifar_100_labels = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]
cifar_10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if cifar_version == 10:
    labels = cifar_10_labels
    n_classes = 10
else:
    labels = cifar_100_labels
    n_classes = 100

class_labels = {i: label for i, label in enumerate(labels)}

#----- Json and Visualization paths here -----#
json_path = 'safetySSLcifar10_session_details.json'
visualization_folder = "Visualization"

if not os.path.exists(json_path):
    raise FileNotFoundError(f"The JSON file '{json_path}' does not exist.")
if os.path.getsize(json_path) == 0:
    raise ValueError(f"The JSON file '{json_path}' is empty.")

#----- Comment in or out depending needed visualizations here -----#
# if cifar_version == 10:
#     plot_confusion_matrix_from_json(json_path, folder=visualization_folder)
# else:
#     plot_confusion_matrix100_from_json(json_path, folder=visualization_folder)

#display_images(json_path, class_labels, folder=visualization_folder)
#plot_accuracy_loss_from_json(json_path, folder=visualization_folder)
# plot_roc_avg(json_path, n_classes, folder=visualization_folder)
# plot_roc(json_path, n_classes, list(class_labels.values()), folder=visualization_folder)
plot_top_confidences(json_path, labels, folder=visualization_folder)
#scores = calculate_scores_from_json(json_path)
#plot_scores_across_epochs(scores, folder=visualization_folder)