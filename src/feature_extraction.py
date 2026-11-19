import torch
from torchvision import models
from torchvision.models.resnet import BasicBlock


weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

model.eval()

print(model)



print(BasicBlock)