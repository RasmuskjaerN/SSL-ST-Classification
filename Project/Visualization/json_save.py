import json
import datetime
import torch
from torch import nn

class TimeHistory:
    def __init__(self):
        self.epochs_timing = []
        self.run_start_time = None
        self.run_end_time = None
        self.epoch_start_time = None

    def on_train_begin(self):
        self.run_start_time = datetime.datetime.now()

    def on_train_end(self):
        self.run_end_time = datetime.datetime.now()

    def on_epoch_begin(self):
        self.epoch_start_time = datetime.datetime.now()

    def on_epoch_end(self):
        self.epoch_end_time = datetime.datetime.now()
        self.epochs_timing.append({
            'start_time': self.epoch_start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': self.epoch_end_time.strftime('%Y-%m-%d %H:%M:%S')
        })

def collect_predictions(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
    return predictions

def save_results_to_json(all_epoch_data, time_callback, batch_size, epochs, learning_rate, file_path='results.json'):
    """
    Saves training results and metadata to a JSON file.

    Args:
    all_epoch_data (list of dicts): A list of dictionaries containing epoch-specific training details.
    time_callback (TimeHistory): Time tracking callback with start and end times of the run.
    batch_size (int): Batch size used during training.
    epochs (int): Number of epochs training was run for.
    learning_rate (float): Learning rate used during training.
    file_path (str, optional): File path to save the JSON results. Defaults to 'results.json'.
    """
    data = {
        'metadata': {
            'run_start_time': time_callback.run_start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'run_end_time': time_callback.run_end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'file_generated_on': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pytorch_version': torch.__version__,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate
        },
        'epoch_details': all_epoch_data
    }

    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Failed to save JSON: {e}")
