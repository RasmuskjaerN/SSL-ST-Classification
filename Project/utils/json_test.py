from json_session_utils import SessionManager
from json_model_details import ModelDetails
from json_system_info import SystemInfo
from json_training_data import TrainingData
from json_training_session import Epoch

# Create instances
session = SessionManager()
model = ModelDetails("Linear Classification", "1.0", 0.01, 32, "SGD")
system = SystemInfo("Local")
training_data = TrainingData("Cifar10", 10000, "Normalization, Resizing")

# Populate session data
session.data['model_details'] = model.get_details()
session.data['system_info'] = system.get_system_info()
session.data['training_data'] = training_data.get_data()

# Simulate training process
epoch = Epoch(1, session.get_current_time(), session.get_current_time(), 0.25, 0.30, 95, 93)
epoch.add_sample("img_001", "cat", "dog", {"cat": 0.1, "dog": 0.9, "mouse": 0.0}, [[255, 0, 0], [0, 255, 0], [0, 0, 255]])
session.data['epochs'].append(epoch.get_epoch_data())

# Finalize and output JSON
session.close_session()
print(session.to_json())
