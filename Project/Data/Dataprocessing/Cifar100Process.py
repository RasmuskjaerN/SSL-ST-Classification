import tarfile
import os
import matplotlib.pyplot as plt

# # Path to your downloaded CIFAR-100 tar file
# tar_file_path = 'D:\SSL-ThesisProject\Project\Data\cifar-100-python.tar.gz'

# # Target directory where you want to extract the files
# extract_to_path = 'D:\SSL-ThesisProject\Project\Data'

# # Extract the tar file
# with tarfile.open(tar_file_path, 'r') as tar:
#     tar.extractall(path=extract_to_path)


import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar100_data(data_dir):
    # Load training data
    train_file = os.path.join(data_dir, 'train')
    train_data_dict = unpickle(train_file)
    train_data = train_data_dict[b'data']
    train_labels = train_data_dict[b'fine_labels']  # Use b'coarse_labels' for the superclasses

    train_data = train_data.reshape((len(train_data), 3, 32, 32)).transpose(0, 2, 3, 1)

    # Load test data
    test_file = os.path.join(data_dir, 'test')
    test_data_dict = unpickle(test_file)
    test_data = test_data_dict[b'data']
    test_labels = test_data_dict[b'fine_labels']  # Use b'coarse_labels' for the superclasses

    test_data = test_data.reshape((len(test_data), 3, 32, 32)).transpose(0, 2, 3, 1)

    # Load meta data for label names
    meta_file = os.path.join(data_dir, 'meta')
    meta_dict = unpickle(meta_file)
    fine_label_names = [label.decode('utf-8') for label in meta_dict[b'fine_label_names']]  # Use b'coarse_label_names' for the superclass names

    return train_data, train_labels, test_data, test_labels, fine_label_names

data_dir = 'D:\SSL-ThesisProject\Project\Data\cifar-100-python'  # Update this path
train_data, train_labels, test_data, test_labels, label_names = load_cifar100_data(data_dir)

def plot_images(images, labels, label_names, num_images=5):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(label_names[labels[i]])
    plt.show()

# Plot a few images from the training set
# plot_images(train_data, train_labels, label_names, 10)