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
import json
import datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a student model using knowledge distillation."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train."
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



    # Transformation pipeline for input data
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        clamp_image  # Ensures all values are within [0, 1]
    ])
    #Change data dir depending on dataset
    # data_dir = "Project/Data/cifar-10-batches-py"
    data_dir = "Project/Data/cifar-100-python/"
    # train_data, train_labels, test_data, test_labels, label_names = CIFAR100Dataset(data_dir)._load_data_cifar100
    train_dataset = CIFAR100Dataset(data_dir=data_dir, transform=transform, train=True)
    test_dataset = CIFAR100Dataset(data_dir=data_dir, transform=transform, train=False)
    # dataset = CIFAR100Dataset(data_dir=data_dir, transform=transform, train=True)



    # Distributed sampler for training data
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)

    # Distributed sampler for validation data
    # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    # Initialize models
    student = Student(num_labels=100).to(rank)
    teacher = Teacher().to(rank)



    student = DDP(student, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    teacher = DDP(teacher, device_ids=[rank], output_device=rank, find_unused_parameters=True)


    # Initialize the optimizer

    optimizer = torch.optim.SGD(student.parameters(), lr=args.learning_rate, momentum=0.9)

    # Load the checkpoint
    checkpoint_path = 'student_model_ddp_epoch_15.pth' #'student_model_ddp_epoch_50.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(rank))
        model_state_dict = checkpoint['model_state_dict']
        
        # Adjust model_state_dict to fit the current model structure or ignore incompatible keys
        student.load_state_dict(model_state_dict, strict=False)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            print("Optimizer state not loaded due to parameter mismatch:", e)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Decays the LR by a factor of 0.1 every 10 epochs


    all_epoch_data = []
    start_epoch = load_checkpoint('student_model_ddp_epoch_13.pth', student, optimizer)



    for epoch in range(start_epoch, args.epochs):
        train_student_with_teacher(
            student_model=student,
            teacher_model=teacher,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            device=rank,  # Device should match rank in DDP
            temperature=2.0,
            alpha=0.5,
            teacher_channels=1024,  # Adjust according to your model's architecture
            student_channels=384,  # Adjust according to your model's architecture
            output_size=100  # Output size should match your dataset's number of classes if classification is involved
        )


        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        save_model(student, optimizer, epoch, f'student_model_ddp_epoch_{epoch+1}.pth', rank)

 

    
    cleanup()
