import torch
from models.modified_resnet import ModifiedResNet18
from models.lm_loss import LMLoss
from data.dataset import load_dataset
from training.trainer import train_model
from configs.hyperparameters import alpha, beta, gamma, delta_v, delta_d, num_classes, num_epochs, learning_rate
from torchvision import transforms

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
train_loader, val_loader, test_loader = load_dataset(train_transform=train_transform, test_transform=test_transform)

# Model, Loss, and Optimizer
model = ModifiedResNet18(num_classes=num_classes).to(device)
criterion = LMLoss(alpha, beta, gamma, delta_v, delta_d)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_losses, val_losses = train_model(model, criterion, optimizer, num_epochs, train_loader, val_loader, device)