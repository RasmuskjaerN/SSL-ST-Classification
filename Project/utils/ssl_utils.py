import os
import numpy as np
import pickle
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
import torch

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class SSLCIFAR10Dataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True, labeled=True, num_labeled=None, validation_split=0.2):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.labeled = labeled
        self.validation_split = validation_split
        self.data = []
        self.labels = []

        self._load_data()
        if self.train:  
            self._train_validation_split()
            if num_labeled is not None:
                self._split_labeled_unlabeled(num_labeled)

    def _load_batch(self, file_path):
        data_dict = unpickle(file_path)
        images = data_dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
        labels = np.array(data_dict[b'labels'])
        return images, labels

    def _load_data(self):
        files_to_load = [f'data_batch_{i}' for i in range(1,6)] if self.train else ['test_batch']
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._load_batch, os.path.join(self.data_dir, file)) for file in files_to_load]
            for future in futures:
                data, labels = future.result()
                self.data.append(data)
                self.labels.extend(labels)
        self.data = np.vstack(self.data)
        self.labels = np.array(self.labels)

    def _train_validation_split(self):
        num_samples = len(self.data)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        num_val = int(np.floor(self.validation_split * num_samples))
        self.validation_indices = indices[:num_val]
        self.train_indices = indices[num_val:]

    def _split_labeled_unlabeled(self, num_labeled):
        if num_labeled > len(self.train_indices):
            raise ValueError("num_labeled is greater than the number of available training samples")
        np.random.shuffle(self.train_indices)
        self.labeled_indices = self.train_indices[:num_labeled]
        self.unlabeled_indices = self.train_indices[num_labeled:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx] if self.labeled else -1
        if self.transform:
            img = self.transform(Image.fromarray(img.astype('uint8')))
        else:
            img = torch.tensor(img.transpose((2, 0, 1)), dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return img, label


class SSLCIFAR100Dataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True, labeled=True, num_labeled=None, validation_split=0.2):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.labeled = labeled
        self.validation_split = validation_split
        self.data = []
        self.labels = []

        self.labeled_indices = []
        self.unlabeled_indices = []
        self.validation_indices = []

        self._load_data_cifar100()

        if self.train:  # Only split into train and validation if dealing with training data
            self._train_validation_split()
            if num_labeled is not None:
                self._split_labeled_unlabeled(num_labeled)

    def _load_data_cifar100(self):
        """Load CIFAR-100 data from files."""
        file_to_load = 'train' if self.train else 'test'
        file_path = os.path.join(self.data_dir, f"{file_to_load}")
        data_dict = unpickle(file_path)
        images = data_dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
        labels = np.array(data_dict[b'fine_labels'])
        self.data = images
        self.labels = labels
        print(f"Data loaded: {len(self.data)} images")

    def _train_validation_split(self):
        """Split the data into training and validation sets."""
        num_samples = len(self.data)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        num_val = int(np.floor(self.validation_split * num_samples))
        self.validation_indices = indices[:num_val]
        self.train_indices = indices[num_val:]
        print(f"Validation split: {len(self.validation_indices)}, Train split: {len(self.train_indices)}")

    def _split_labeled_unlabeled(self, num_labeled):
        """Split training data into labeled and unlabeled datasets."""
        if num_labeled > len(self.train_indices):
            raise ValueError("num_labeled is greater than the number of available training samples")
        np.random.shuffle(self.train_indices)
        self.labeled_indices = self.train_indices[:num_labeled]
        self.unlabeled_indices = self.train_indices[num_labeled:]
        print(f"Labeled: {len(self.labeled_indices)}, Unlabeled: {len(self.unlabeled_indices)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx] if self.labeled else -1
        if self.transform:
            img = self.transform(Image.fromarray(img.astype('uint8')))
        else:
            img = torch.tensor(img.transpose((2, 0, 1)), dtype=torch.float32)  # Ensure it's a tensor and match the expected shape
        label = torch.tensor(label, dtype=torch.long)
        return img, label

      
    # def __getitem__(self, idx):
    #     # Handle case where idx is an array-like object
    #     if isinstance(idx, (list, tuple, np.ndarray)):
    #         if len(idx) == 1:
    #             idx = idx[0]
    #         else:
    #             raise ValueError("Index array must be of length 1 for single item fetch.")

    #     img = self.data[idx]
    #     label = self.labels[idx] if self.labeled else -1
    #     img = Image.fromarray(img.astype(np.uint8))

    #     if self.transform:
    #         img = self.transform(img)

    #     if self.labeled:
    #         return img, label
    #     else:
    #         return img