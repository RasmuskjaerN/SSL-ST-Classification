import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
import argparse
import os


from torch.optim.lr_scheduler import StepLR 

from SSL_Student_Teacher.student import Student
from SSL_Student_Teacher.teacher import Teacher
from Train.student_train import train_student_with_teacher
from utils.data_utils import CIFAR10Dataset, CIFAR100Dataset
from Models.linear_classification_model import SimpleLinearModel, SimpleLinearModelConfig
from Train.linear_classification_training import train_classification_model, validate_classification_model, combined_train_validate
from Visualization.json_save import TimeHistory, save_results_to_json, collect_predictions
import ujson as json  # UltraJSON, a fast JSON encoder/decoder
import datetime

from utils.json_session_utils import SessionManager
from utils.json_model_details import ModelDetails
from utils.json_system_info import SystemInfo
from utils.json_training_data import TrainingData
from utils.json_training_session import Epoch



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a student model using knowledge distillation."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Num workers for training"
    )
    # Add more arguments as needed
    return parser.parse_args()

def setup(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')  # Default to localhost if not set
    os.environ.setdefault('MASTER_PORT', '12355')  # Use an arbitrary free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    


def cleanup():
    dist.destroy_process_group()

def save_model(model, optimizer, epoch, filepath, rank):
    """
    Save the model and optimizer state dictionaries.

    Args:
        model (torch.nn.Module): The model whose state to save.
        optimizer (torch.optim.Optimizer): The optimizer whose state to save.
        filepath (str): Path to save the checkpoint file.
        rank (int): The rank of the current process in distributed training.
    """
    if rank == 0:  # Ensure only the master process saves the checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, filepath)

def load_checkpoint(filepath, model, optimizer):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            epoch = checkpoint.get('epoch', 0) + 1  # Default to 0 if not found
            print(f"Resuming from epoch {epoch}")
        return epoch
    else:
        print(f"No checkpoint found at {filepath}, starting from scratch.")
        return 0

def clamp_image(img):
    return img.clamp(0, 1)

def main_worker(rank, world_size, args):
    
    args = parse_args()  # Get the runtime arguments
    setup(rank, world_size)  # Setup distributed environment


    dist.barrier()  # Synchronize all processes before starting

    time_history = TimeHistory()
    time_history.on_train_begin()

    # Instantiate session, system info, and model details managers
    session_manager = SessionManager()
    system_info = SystemInfo("Cloud")  # Or "Local", based on your environment
    model_details = ModelDetails("SimpleLinearModel", "1.0", args.learning_rate, args.batch_size, "SGD")

    # Store initial configuration in session data
    session_manager.data['system_info'] = system_info.get_system_info()
    session_manager.data['model_details'] = model_details.get_details()

    # Transformation pipeline for input data
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        clamp_image  # Ensures all values are within [0, 1]
    ])
    #Change data dir depending on dataset
    data_dir = "Project/Data/cifar-10-batches-py"
    #data_dir = "Project/Data/cifar-100-python/"

    train_dataset = CIFAR10Dataset(data_dir=data_dir, transform=transform, train=True)
    test_dataset = CIFAR10Dataset(data_dir=data_dir, transform=transform, train=False)
    #dataset = CIFAR100Dataset(data_dir=data_dir, transform=transform, train=True)


    dataset = train_dataset
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Distributed sampler for training data
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    # Distributed sampler for validation data
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    # Inside main_worker or similar initialization section
    training_data = TrainingData("Cifar100", len(dataset), "Normalization, Resizing")
    session_manager.data['training_data'] = training_data.get_data()

    # Initialize models
    student = Student(num_labels=100).to(rank)
    teacher = Teacher().to(rank)

    config = SimpleLinearModelConfig(student_model_checkpoint='facebook/deit-small-distilled-patch16-224', num_classes=10) #change depending on number of classes
    config.save_pretrained('./model_config_directory/linear/') 
    loaded_config = SimpleLinearModelConfig.from_pretrained('./model_config_directory/linear/')
    # # Initialize the model with configuration and optional state dictionary   
    model_state_dict = 'student_model_ddp_epoch_20.pth'
    linear_model = SimpleLinearModel(config=loaded_config, state_dict_path=model_state_dict).to(rank)

    for param in student.parameters():
        param.requires_grad = False

    model = DDP(linear_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)


    # Initialize the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)


    # Load the checkpoint
    checkpoint_path = 'student_model_ddp_epochr_50.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(rank))
        print(checkpoint.keys())
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        
        # Load model state
        model.load_state_dict(model_state_dict, strict=False)

        # Attempt to load optimizer state
        current_param_ids = {id(p): i for i, p in enumerate(optimizer.param_groups[0]['params'])}
        new_optimizer_state_dict = {'state': {}, 'param_groups': optimizer.state_dict()['param_groups']}

        # Adjust the parameter groups to match the current optimizer setup
        for key, value in optimizer_state_dict['state'].items():
            if key in current_param_ids:
                new_optimizer_state_dict['state'][current_param_ids[key]] = value

        try:
            optimizer.load_state_dict(new_optimizer_state_dict)
        except ValueError as e:
            print("Optimizer state not loaded due to parameter mismatch:", e)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Decays the LR by a factor of 0.1 every 10 epochs



    all_epoch_data = []


    for epoch in range(args.epochs):
            time_history.on_epoch_begin()
            dist.barrier()  # Synchronize all processes
            train_loader.sampler.set_epoch(epoch)

            epoch_results = combined_train_validate(model, train_loader, val_loader, optimizer, epoch, rank)
            scheduler.step()

            
            epoch_data = Epoch(epoch, session_manager.get_current_time(), session_manager.get_current_time(), epoch_results['train_loss'], epoch_results['val_loss'], epoch_results['train_accuracy'], epoch_results['val_accuracy'])
            if dist.get_rank() == 0:
                    
                print(len(epoch_results['val_image_pixels'])) # 25
                print(len(epoch_results['val_predictions'])) # 10000

                for i in range(len(epoch_results['val_predictions'])):
                    true_labels = epoch_results['val_true_labels'][i]
                    predictions = epoch_results['val_predictions'][i]
                    confidences = epoch_results['val_confidences'][i]

                    # Assuming 'epoch_data' is some kind of container for samples
                    if i >= (len(epoch_results['val_predictions'])//4) - len(epoch_results['val_image_pixels']) and i < (len(epoch_results['val_predictions'])//4): #2475-2500 i < len(epoch_results['val_image_pixels']):
                        # Assuming 'epoch_data.add_sample' adds a sample to 'epoch_data'
                        print(i - ((len(epoch_results['val_predictions'])//4) - len(epoch_results['val_image_pixels'])))
                        epoch_data.add_sample(f"img_{i}", true_labels, predictions, confidences, epoch_results['val_image_pixels'][i - ((len(epoch_results['val_predictions'])//4) - len(epoch_results['val_image_pixels']))])
                    else:
                        epoch_data.add_sample(f"img_{i}", true_labels, predictions, confidences, "")

                session_manager.data['epochs'].append(epoch_data.get_epoch_data())


                if rank == 0:  # Master node logs the detailed results
                    print(f"Epoch {epoch+1} completed. Training Loss: {epoch_results['train_loss']:.4f}, Validation Accuracy: {epoch_results['val_accuracy']:.2f}%")
                    save_model(model, optimizer, epoch, f'student_model_ddp_epoch_{epoch+1}.pth', rank)

    if rank == 0:

        with open('cifar100_session_details.json', 'w') as f:
            json.dump(session_manager.data, f, indent=4)


    session_manager.close_session()
    cleanup()
