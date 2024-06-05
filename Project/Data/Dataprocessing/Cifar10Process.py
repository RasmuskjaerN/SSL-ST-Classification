# from zipfile import ZipFile

# with ZipFile("D:\SSL-ThesisProject\cifar10-python.zip") as zObject:
#     zObject.extractall(path='Project\Data')
import pickle
import numpy as np

import matplotlib.pyplot as plt


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


batch_files = [
    "Project/Data/cifar-10-batches-py/data_batch_1",
    "Project/Data/cifar-10-batches-py/data_batch_2",
    "Project/Data/cifar-10-batches-py/data_batch_3",
    "Project/Data/cifar-10-batches-py/data_batch_4",
    "Project/Data/cifar-10-batches-py/data_batch_5",
]

all_images = []
all_labels = []
for batch_file in batch_files:
    batch_data = unpickle(batch_file)

    all_images.append(batch_data[b"data"])
    all_labels += batch_data[b"labels"]

all_images = np.concatenate(all_images)
all_labels = np.array(all_labels)

# Map of numeric labels to class names
label_names = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


# Load the dataset
image_index = 255

# Reshape and transpose the first image
first_image = all_images[image_index].reshape(3, 32, 32).transpose(1, 2, 0)

# Convert the first label to its string representation
first_label_string = label_names[all_labels[image_index]]

# Plot the first image with its label
print(first_label_string)
plt.imshow(first_image)
plt.title(f"Label: {first_label_string}")
plt.show()
