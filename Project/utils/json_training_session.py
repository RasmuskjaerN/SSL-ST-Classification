# training_session.py

class Epoch:
    """Record and manage details of an individual training epoch."""
    def __init__(self, epoch_number, start_time, end_time, training_loss, validation_loss, training_accuracy, validation_accuracy):
        self.epoch_number = epoch_number
        self.start_time = start_time
        self.end_time = end_time
        self.metrics = {
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "training_accuracy": training_accuracy,
            "validation_accuracy": validation_accuracy
        }
        self.samples = []

    def add_sample(self, image_id, label, prediction, confidences, pixels):
        """Add a sample prediction to the epoch."""
        self.samples.append({
            "image_id": image_id,
            "label": label,
            "prediction": prediction,
            "confidences": confidences,
            "pixels": pixels
        })

    def get_epoch_data(self):
        """Return the epoch data as a dictionary."""
        return {
            "epoch_number": self.epoch_number,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metrics": self.metrics,
            "samples": self.samples
        }
