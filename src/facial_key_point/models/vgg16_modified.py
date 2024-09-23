import torch
import torch.nn as nn
from torchvision import models

def get_model(device): 
  model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
  # print(vgg16)
  for layers in model.parameters():
    layers.requires_grad = False #no gradient descent work, layers donot optimize

  model.avgpool = nn.Sequential(
    nn.Conv2d(512, 512, 3),
    nn.MaxPool2d(2),
    nn.Flatten()
  )
  model.classifier=nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(512, 136),
    nn.Sigmoid() #sigmoid/softmax:sigmoid every point probability 0 to 1
  )
  # print(model)
  return model.to(device=device)

if __name__=="__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = get_model(device=device)
  print(model)


