import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
from torch.distributed import get_rank, get_world_size
import logging
import gc
import os
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO)

def cleanup():
    import torch.multiprocessing as mp
    mp.get_context('spawn').set_start_method('spawn', force=True)
    torch.cuda.empty_cache()
    gc.collect()

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"Rank {get_rank()} Memory Usage: RSS={mem_info.rss / 1024 ** 2:.2f} MB, VMS={mem_info.vms / 1024 ** 2:.2f} MB")

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def combined_train_validate(model, train_loader, val_loader, unlabeled_loader, optimizer, epoch, device, pseudo_label_threshold=0.95, scaler=None):
    """
    Function to perform combined training and validation with pseudo-labeling.

    Args:
        model (nn.Module): The model to train and validate.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        unlabeled_loader (DataLoader): DataLoader for unlabeled data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epoch (int): Current epoch number.
        device (torch.device): Device to run the model on (CPU or GPU).
        pseudo_label_threshold (float): Confidence threshold for pseudo-labeling.
        scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision training.
    
    Returns:
        dict: Dictionary containing training and validation metrics.
    """
    classification_loss_fn = nn.CrossEntropyLoss().to(device)
    model.to(device)
    combined_loader = None

    # Print initial sizes
    if get_rank() == 0:
        logging.info(f'Initial train loader size: {len(train_loader.dataset)}')
        logging.info(f'Initial validation loader size: {len(val_loader.dataset)}')
        logging.info(f'Initial unlabeled loader size: {len(unlabeled_loader.dataset)}')

    try:
        # Training Phase
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}, Training", disable=dist.get_rank() != 0):
            try:
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    outputs = model(inputs)
                    loss = classification_loss_fn(outputs, labels)

                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_train_loss += loss.item()
                _, predicted = outputs.max(1)
                total_train_correct += predicted.eq(labels).sum().item()
                total_train_samples += labels.size(0)
            except Exception as e:
                logging.error(f"Training error: {e}")
                continue

        # Generate pseudo-labels for unlabeled data
        model.eval()
        pseudo_labels = []
        pseudo_data = []

        with torch.no_grad():
            for inputs, _ in tqdm(unlabeled_loader, desc=f"Epoch {epoch+1}, Pseudo-Labeling", disable=dist.get_rank() != 0):
                try:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidences, predicted = probabilities.max(1)
                    high_confidence_mask = confidences > pseudo_label_threshold
                    pseudo_labels.extend(predicted[high_confidence_mask].cpu().tolist())
                    pseudo_data.extend(inputs[high_confidence_mask].cpu())
                except Exception as e:
                    logging.error(f"Pseudo-labeling error: {e}")
                    continue
        
        logging.info(f"Rank {get_rank()}: Pseudo-labels generated: {len(pseudo_labels)}")
        torch.cuda.empty_cache()
        logging.info(f"Rank {get_rank()}: Cleared CUDA cache before pseudo-labeling")

        if pseudo_labels:
            try:
                pseudo_labels_tensor = torch.tensor(pseudo_labels, dtype=torch.long).to(device)
                pseudo_data_tensor = torch.stack(pseudo_data).to(device)

                local_size = torch.tensor([len(pseudo_labels_tensor)], dtype=torch.long).to(device)
                all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(get_world_size())]
                dist.all_gather(all_sizes, local_size)
                all_sizes = [size.item() for size in all_sizes]

                total_size = sum(all_sizes)
                max_size = max(all_sizes)

                logging.info(f"Rank {get_rank()}: All sizes gathered: {all_sizes}")

                # Pad the tensors to the maximum size
                if len(pseudo_labels_tensor) < max_size:
                    padding_labels = torch.full((max_size - len(pseudo_labels_tensor),), -1, dtype=torch.long, device=device)
                    pseudo_labels_tensor = torch.cat([pseudo_labels_tensor, padding_labels])

                    padding_data = torch.zeros((max_size - pseudo_data_tensor.size(0),) + pseudo_data_tensor.size()[1:], dtype=pseudo_data_tensor.dtype, device=device)
                    pseudo_data_tensor = torch.cat([pseudo_data_tensor, padding_data])

                # Allocate memory for gathered tensors based on total size
                gathered_pseudo_labels_list = [torch.zeros(max_size, dtype=pseudo_labels_tensor.dtype, device=device) for _ in range(get_world_size())]
                gathered_pseudo_data_list = [torch.zeros((max_size,) + pseudo_data_tensor.size()[1:], dtype=pseudo_data_tensor.dtype, device=device) for _ in range(get_world_size())]

                # Use all_gather to gather data from all GPUs
                dist.all_gather(gathered_pseudo_labels_list, pseudo_labels_tensor)
                dist.all_gather(gathered_pseudo_data_list, pseudo_data_tensor)

                logging.info(f"Rank {get_rank()}: Gathered pseudo-labels and data")

                # Concatenate gathered data from all GPUs
                gathered_pseudo_labels = torch.cat(gathered_pseudo_labels_list).cpu().numpy()
                gathered_pseudo_data = torch.cat(gathered_pseudo_data_list).cpu()

                # Remove padding
                valid_indices = gathered_pseudo_labels != -1
                all_pseudo_labels = gathered_pseudo_labels[valid_indices]
                all_pseudo_data = gathered_pseudo_data[valid_indices]

                logging.info(f"Rank {get_rank()}: Size of all pseudo-labels: {len(all_pseudo_labels)}")
                logging.info(f"Rank {get_rank()}: Size of all pseudo-data: {len(all_pseudo_data)}")

                pseudo_dataset = TensorDataset(all_pseudo_data, torch.tensor(all_pseudo_labels))
                combined_dataset = ConcatDataset([train_loader.dataset, pseudo_dataset])
                combined_sampler = DistributedSampler(combined_dataset, num_replicas=get_world_size(), rank=get_rank())
                combined_loader = DataLoader(combined_dataset, batch_size=train_loader.batch_size, sampler=combined_sampler)

                logging.info(f"Rank {get_rank()}: Combined dataset created with size {len(combined_loader.dataset)}")

            except Exception as e:
                logging.error(f"Pseudo-labeling gathering error: {e}")

        # Training Phase with Combined Data
        if combined_loader:
            model.train()
            total_train_loss = 0
            total_train_correct = 0
            total_train_samples = 0

            for inputs, labels in tqdm(combined_loader, desc=f"Epoch {epoch+1}, Training Combined", disable=dist.get_rank() != 0):
                try:
                    inputs, labels = inputs.to(device), labels.to(device)

                    with torch.cuda.amp.autocast(enabled=scaler is not None):
                        outputs = model(inputs)
                        loss = classification_loss_fn(outputs, labels)

                    optimizer.zero_grad()
                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    total_train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_train_correct += predicted.eq(labels).sum().item()
                    total_train_samples += labels.size(0)
                except Exception as e:
                    logging.error(f"Training combined error: {e}")
                    continue

            train_loss = total_train_loss / len(combined_loader)
            train_accuracy = total_train_correct / total_train_samples
        else:
            train_loss = total_train_loss / len(train_loader)
            train_accuracy = total_train_correct / total_train_samples

        # Validation Phase
        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        total_val_samples = 0

        all_val_predictions = []
        all_val_confidences = []
        val_true_labels = []
        val_image_pixels = []

        with torch.no_grad():
             for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}, Validation", disable=dist.get_rank() != 0):
                try:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = classification_loss_fn(outputs, labels)

                    total_val_loss += loss.item()
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    _, predicted = probabilities.max(1)

                    total_val_correct += predicted.eq(labels).sum().item()
                    total_val_samples += labels.size(0)

                    all_val_predictions.append(predicted.cpu())
                    all_val_confidences.append(probabilities.cpu())
                    val_true_labels.append(labels.cpu())
                    val_image_pixels.append(inputs.cpu().permute(0, 2, 3, 1))
                except Exception as e:
                    logging.error(f"Validation error: {e}")
                    continue

        logging.info(f"Rank {get_rank()}: Completed validation phase")

        if all_val_predictions:
            logging.info("will try logging")
            all_val_predictions = torch.cat(all_val_predictions)
            logging.info("Logged all_val_predictions")
            all_val_confidences = torch.cat(all_val_confidences)
            logging.info("Logged all_val_confidences")
            val_true_labels = torch.cat(val_true_labels)
            logging.info("Logged true labels")
            if epoch == 0 or epoch == 4:
                val_image_pixels = torch.cat(val_image_pixels).numpy().tolist()
                logging.info("Logged pixels")
                val_image_pixels = val_image_pixels[-25:]
                logging.info("Logged some stuff")
            else:
                logging.info("Epoch Does not need Pixels")
        else:
            all_val_predictions = torch.tensor([], dtype=torch.long)
            logging.info("Damn it was empty")
            all_val_confidences = torch.tensor([], dtype=torch.float)
            logging.info("Damn it was empty")
            val_true_labels = torch.tensor([], dtype=torch.long)
            logging.info("Damn it was empty")
            val_image_pixels = torch.tensor([], dtype=torch.long)
            logging.info("Damn it was empty")

        try:
            gathered_predictions = [torch.zeros_like(all_val_predictions).to(device) for _ in range(get_world_size())]
            logging.info(f"1")
            gathered_confidences = [torch.zeros_like(all_val_confidences).to(device) for _ in range(get_world_size())]
            logging.info(f"2")
            gathered_val_true_labels = [torch.zeros_like(val_true_labels).to(device) for _ in range(get_world_size())]
            logging.info(f"3")

            dist.all_gather(gathered_predictions, all_val_predictions.to(device))
            logging.info(f"4")
            dist.all_gather(gathered_confidences, all_val_confidences.to(device))
            logging.info(f"5")
            dist.all_gather(gathered_val_true_labels, val_true_labels.to(device))


            logging.info(f"Rank {get_rank()}: Completed all_gather for validation data")

            if get_rank() == 0:
                all_val_predictions = torch.cat(gathered_predictions)
                logging.info(f"6")
                all_val_confidences = torch.cat(gathered_confidences)
                logging.info(f"7")
                val_true_labels = torch.cat(gathered_val_true_labels)


                logging.info(f'End of Epoch {epoch+1}. Training Loss: {total_train_loss / len(train_loader):.4f}, '
                             f'Training Accuracy: {100. * total_train_correct / total_train_samples:.2f}%, '
                             f'Validation Loss: {total_val_loss / len(val_loader):.4f}, '
                             f'Validation Accuracy: {100. * total_val_correct / total_val_samples:.2f}%')
        except Exception as e:
            logging.error(f"Error during validation data gathering: {e}")

        log_memory_usage()
        
        return {
            "train_loss": total_train_loss / len(train_loader) if combined_loader is None else total_train_loss / len(combined_loader),
            "train_accuracy": 100. * total_train_correct / total_train_samples,
            "val_loss": total_val_loss / len(val_loader),
            "val_accuracy": 100. * total_val_correct / total_val_samples,
            "val_predictions": all_val_predictions.tolist(),
            "val_confidences": all_val_confidences.tolist(),
            "val_true_labels": val_true_labels.tolist(),
            "val_image_pixels": val_image_pixels
        }
        #Ensure to cleanup
    except Exception as e:
        logging.error(f"Error in combined training and validation process: {e}")
        cleanup()  
        log_memory_usage()
        return {
            "train_loss": None,
            "train_accuracy": None,
            "val_loss": None,
            "val_accuracy": None,
            "val_predictions": [],
            "val_confidences": [],
            "val_true_labels": [],
            "val_image_pixels": []
        }
