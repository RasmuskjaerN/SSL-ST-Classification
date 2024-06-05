# model_details.py

class ModelDetails:
    """Manage model information and hyperparameters."""
    def __init__(self, name, version, learning_rate, batch_size, optimizer):
        self.model_name = name
        self.model_version = version
        self.hyperparameters = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "optimizer": optimizer
        }
    
    def get_details(self):
        """Return the model details as a dictionary."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "hyperparameters": self.hyperparameters
        }
