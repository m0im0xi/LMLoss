import torch
import torch.nn as nn
from torchvision import models

class ModifiedResNet18(nn.Module):
  def __init__(self, model, num_classes):
    super(ModifiedResNet18, self).__init__()
    self.features = nn.Sequential(*list(model.children())[:-1])
    self.fc = nn.Linear(model.fc.in_features, num_classes)

  def forward(self, x):
    x = self.features(x)
    features = torch.flatten(x, 1)
    outputs = self.fc(features)
    return features, outputs