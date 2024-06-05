import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
import argparse
import torch.distributed as dist
import logging
import os
import gc
import psutil
import numpy as np
from tqdm import tqdm
import ujson as json
from SSL_Student_Teacher.student import Student
from SSL_Student_Teacher.teacher import Teacher
from Models.linear_classification_model import SimpleLinearModel, SimpleLinearModelConfig
from Train.ssl_train import combined_train_validate
from utils.json_session_utils import SessionManager
from utils.json_model_details import ModelDetails
from utils.json_system_info import SystemInfo
from utils.json_training_data import TrainingData
from utils.json_training_session import Epoch
from utils.ssl_utils import SSLCIFAR10Dataset

# Setup logging
logging.basicConfig(level=logging.INFO)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model using semi-supervised learning.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading.")
    parser.add_argument("--num_labeled", type=int, default=20000, help="Number of labeled samples to use.")
    parser.add_argument("--port", type=int, default=12355, help="Port number for distributed training.")
    return parser.parse_args()


def print_system_resources():
    logging.info(f"Number of physical CPUs: {psutil.cpu_count(logical=False)}")
    logging.info(f"Number of logical CPUs: {psutil.cpu_count(logical=True)}")
    logging.info(f"CPU frequency: {psutil.cpu_freq().current:.2f} MHz")
    logging.info(f"CPU usage per core: {psutil.cpu_percent(percpu=True)}")
    logging.info(f"Total CPU usage: {psutil.cpu_percent()}%")
    virtual_mem = psutil.virtual_memory()
    logging.info(f"Total memory: {virtual_mem.total / (1024 ** 3):.2f} GB")
    logging.info(f"Available memory: {virtual_mem.available / (1024 ** 3):.2f} GB")
    logging.info(f"Used memory: {virtual_mem.used / (1024 ** 3):.2f} GB")
    logging.info(f"Memory usage: {virtual_mem.percent}%")
    if torch.cuda.is_available():
        logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logging.info(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / (1024 ** 3):.2f} GB")
            logging.info(f"  Memory Cached: {torch.cuda.memory_reserved(i) / (1024 ** 3):.2f} GB")
    else:
        logging.info("No GPUs available.")

def cleanup():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    torch.cuda.empty_cache()
    gc.collect()

def clamp_image(img):
    return img.clamp(0, 1)

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

def save_model(model, optimizer, epoch, filepath, rank):
    if rank == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, filepath)

def create_half_subset(full_indices):
    np.random.shuffle(full_indices)
    return full_indices[:len(full_indices) // 2]

def check_dataloader(dataloader):
    for data in dataloader:
        pass  # Implement any checks you need on the dataloader

def main_worker(rank, world_size, args):
    cleanup()
    setup(rank, world_size, args.port)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    dist.barrier()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    session_manager = SessionManager()
    system_info = SystemInfo("Cloud")
    model_details = ModelDetails("SimpleLinearModel", "1.0", args.learning_rate, args.batch_size, "SGD")

    session_manager.data['system_info'] = system_info.get_system_info()
    session_manager.data['model_details'] = model_details.get_details()

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        clamp_image
    ])

    data_dir = 'Project/Data/cifar-10-batches-py'
    full_dataset = SSLCIFAR10Dataset(data_dir, transform=transform, train=True, labeled=True, num_labeled=args.num_labeled, validation_split=0.2)

    labeled_half_indices = create_half_subset(full_dataset.labeled_indices)
    unlabeled_half_indices = create_half_subset(full_dataset.unlabeled_indices)
    validation_half_indices = create_half_subset(full_dataset.validation_indices)

    labeled_dataset = Subset(full_dataset, labeled_half_indices)
    unlabeled_dataset = Subset(full_dataset, unlabeled_half_indices)
    validation_dataset = Subset(full_dataset, validation_half_indices)

    labeled_sampler = DistributedSampler(labeled_dataset, num_replicas=world_size, rank=rank)
    unlabeled_sampler = DistributedSampler(unlabeled_dataset, num_replicas=world_size, rank=rank)
    validation_sampler = DistributedSampler(validation_dataset, num_replicas=world_size, rank=rank)

    labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, sampler=labeled_sampler, num_workers=args.num_workers, pin_memory=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, sampler=unlabeled_sampler, num_workers=args.num_workers, pin_memory=True, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, sampler=validation_sampler, num_workers=args.num_workers, pin_memory=True)

    logger.info(f"Labeled dataset size: {len(labeled_loader.dataset)}")
    logger.info(f"Unlabeled dataset size: {len(unlabeled_loader.dataset)}")
    logger.info(f"Validation dataset size: {len(validation_loader.dataset)}")

    logger.info(f"Completed dataloader for labeled_loader with the size: {len(labeled_loader)}")
    logger.info(f"Completed dataloader for unlabeled_loader with the size: {len(unlabeled_loader)}")
    logger.info(f"Completed dataloader for validation_loader with the size: {len(validation_loader)}")

    check_dataloader(labeled_loader)
    check_dataloader(unlabeled_loader)
    check_dataloader(validation_loader)

    training_data = TrainingData("Cifar10", len(labeled_loader.dataset), "Normalization, Resizing")
    session_manager.data['training_data'] = training_data.get_data()

    student = Student(num_labels=10).to(rank)
    teacher = Teacher().to(rank)

    config = SimpleLinearModelConfig(student_model_checkpoint='facebook/deit-small-distilled-patch16-224', num_classes=10)
    config.save_pretrained('./model_config_directory/linear/') 
    loaded_config = SimpleLinearModelConfig.from_pretrained('./model_config_directory/linear/')
    model_state_dict = 'student_model_ddp_epochr_50.pth'
    linear_model = SimpleLinearModel(config=loaded_config, state_dict_path=model_state_dict).to(rank)

    for param in student.parameters():
        param.requires_grad = False
    for param in teacher.parameters():
        param.requires_grad = False

    model = DDP(linear_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    checkpoint_path = 'student_model_ddp_epochr_50.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(rank))
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']

        model.load_state_dict(model_state_dict, strict=False)
        current_param_ids = {id(p): i for i, p in enumerate(optimizer.param_groups[0]['params'])}
        new_optimizer_state_dict = {'state': {}, 'param_groups': optimizer.state_dict()['param_groups']}

        for key, value in optimizer_state_dict['state'].items():
            if key in current_param_ids:
                new_optimizer_state_dict['state'][current_param_ids[key]] = value

        try:
            optimizer.load_state_dict(new_optimizer_state_dict)
        except ValueError as e:
            logger.warning("Optimizer state not loaded due to parameter mismatch:", e)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch+1}")
        print_system_resources()
        
        dist.barrier()  # Synchronize all processes
        labeled_loader.sampler.set_epoch(epoch)
        unlabeled_loader.sampler.set_epoch(epoch)

        epoch_results = combined_train_validate(model, labeled_loader, validation_loader, unlabeled_loader, optimizer, epoch, device=rank, pseudo_label_threshold=0.93, scaler=None)
        scheduler.step()

        epoch_data = Epoch(epoch, session_manager.get_current_time(), session_manager.get_current_time(), epoch_results['train_loss'], epoch_results['val_loss'], epoch_results['train_accuracy'], epoch_results['val_accuracy'])
        if dist.get_rank() == 0:
            print(len(epoch_results['val_image_pixels']))
            print(len(epoch_results['val_predictions']))
            for i in range(len(epoch_results['val_predictions'])):
                true_labels = epoch_results['val_true_labels'][i]
                predictions = epoch_results['val_predictions'][i]
                confidences = epoch_results['val_confidences'][i]
                if epoch_results['val_image_pixels']:
                    if i >= (len(epoch_results['val_predictions']) // 4) - len(epoch_results['val_image_pixels']) and i < (len(epoch_results['val_predictions']) // 4):
                        print(i - ((len(epoch_results['val_predictions']) // 4) - len(epoch_results['val_image_pixels'])))
                        epoch_data.add_sample(f"img_{i}", true_labels, predictions, confidences, epoch_results['val_image_pixels'][i - ((len(epoch_results['val_predictions']) // 4) - len(epoch_results['val_image_pixels']))])
                    else:
                        epoch_data.add_sample(f"img_{i}", true_labels, predictions, confidences, "")
                else:
                    epoch_data.add_sample(f"img_{i}", true_labels, predictions, confidences, "")

            session_manager.data['epochs'].append(epoch_data.get_epoch_data())
            if rank == 0:
                logger.info(f"Epoch {epoch + 1} completed. Training Loss: {epoch_results['train_loss']:.4f}, Validation Accuracy: {epoch_results['val_accuracy']:.2f}%")
                save_model(model, optimizer, epoch, f'student_model_ddp_epoch_{epoch + 1}.pth', rank)

    if rank == 0:
        with open('SSLcifar10_session_details.json', 'w') as f:
            json.dump(session_manager.data, f, indent=4)

    session_manager.close_session()
    cleanup()