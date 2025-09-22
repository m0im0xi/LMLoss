import torch
import torch.nn as nn
import torch.nn.functional as F

class LMLoss(nn.Module):
  def __init__(self, alpha, beta, gamma, delta_v, delta_d):
    super(LMLoss, self).__init__()
    self.delta_v = delta_v
    self.delta_d = delta_d
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma

  def _distance(self, point, weight, bias):
    numerator = torch.abs(torch.dot(weight, point) + bias)
    denominator = torch.norm(weight)
    return numerator / denominator

  def _computeCentroids(self, features, labels, unique_labels, momentum=0.99):
    if not hasattr(self, 'centroids'):
      self.centroids = torch.zeros((len(unique_labels), features.size(1)), device=features.device)
      
    for label in unique_labels:
      mask = (labels == label)
      class_features = features[mask]
      new_centroid = class_features.mean(dim=0) 
      self.centroids[label] = momentum * self.centroids[label] + (1 - momentum) * new_centroid
      
    return self.centroids

  def _compactLoss(self, features, labels, unique_labels, centroids):
    # Calculate the compactness loss
    loss = 0.0
    for i, label in enumerate(unique_labels):
      mask = (labels == label)
      class_features = features[mask]
      centroid = centroids[i]
      distance = torch.norm(class_features - centroid, dim=1)
      hinge_loss = F.relu(distance - self.delta_v)
      per_class_mean = torch.mean(hinge_loss ** 2) * class_features.size(0)
      loss += per_class_mean
    return loss / len(unique_labels)

  def _marginLoss(self, features, labels, weights, biases, unique_labels, centroids):
    # Calculate the margin loss
    loss = 0.0
    for i in range(len(unique_labels)):
      for j in range(len(unique_labels)):
        if i != j:
          weight = weights[i] - weights[j]
          bias = biases[i] - biases[j]
          centroid = centroids[i]
          distance = self._distance(centroid, weight, bias)
          sign = torch.sign(torch.dot(weight, centroid) + bias)
          hinge_loss = F.relu(self.delta_d + distance * sign)
          loss += hinge_loss
    return loss / len(unique_labels)

  def _regLoss(self, centroids):
      loss = torch.mean(torch.norm(centroids, dim=1))
      return loss

  def forward(self, features, labels, weights, biases):
      unique_labels = torch.unique(labels)
      centroids = self._computeCentroids(features, labels, unique_labels).detach()
      compact_loss = self._compactLoss(features, labels, unique_labels, centroids)
      margin_loss = self._marginLoss(features, labels, weights, biases, unique_labels, centroids)
      reg_loss = self._regLoss(centroids)
      total_loss = self.alpha * compact_loss + self.beta * margin_loss + self.gamma * reg_loss

      return total_loss
