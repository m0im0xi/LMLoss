from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_dataset(root='./data', train_transform=None, test_transform=None):
    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, val_loader, test_loader