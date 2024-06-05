import torch
import torchvision
import torchvision.transforms as transforms

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors and normalize pixel values
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std for 3 channels
])

# Load CIFAR-100 dataset
trainset = torchvision.datasets.CIFAR100(root='Project\Data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testset = torchvision.datasets.CIFAR100(root='Project\Data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
