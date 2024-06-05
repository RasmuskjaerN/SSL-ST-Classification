# training_data.py

class TrainingData:
    """Manage training dataset details."""
    def __init__(self, dataset_name, num_images, preprocessing):
        self.dataset = dataset_name
        self.num_images = num_images
        self.preprocessing = preprocessing
    
    def get_data(self):
        """Return the training data information."""
        return {
            "dataset": self.dataset,
            "num_images": self.num_images,
            "preprocessing": self.preprocessing
        }
