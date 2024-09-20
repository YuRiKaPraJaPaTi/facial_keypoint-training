from tqdm import tqdm
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

def train_batch(imgs, kps, model, criterion, optimizer):
  model.train()
  optimizer.zero_grad()

  #forward pass
  kps_pred=model(imgs)
  loss=criterion(kps_pred, kps)

  #backward pass
  loss.backward()
  optimizer.step()

  return loss

@torch.no_grad()
def validation_batch(imgs, kps, model, criterion):
  model.eval()

  kps_pred = model(imgs)
  loss = criterion(kps_pred, kps)
  return loss

##training
def train(n_epoch, train_dataloader, test_dataloader, model, criterion, optimizer):
  train_loss=[]
  test_loss=[]

  for epoch in range(1, n_epoch+1):
    epoch_train_loss, epoch_test_loss = 0,0

    #train
    for images, kps in tqdm(train_dataloader, desc=f'Training {epoch} of {n_epoch}'):
      #images, kps
      loss=train_batch(images, kps, model, criterion, optimizer)
      epoch_train_loss += loss.item()
    epoch_train_loss /= len(train_dataloader)
    train_loss.append(epoch_train_loss)

    #validation
    for images, kps in tqdm(test_dataloader, desc=f'Validation {epoch} of {n_epoch}'):
      loss=validation_batch(images, kps, model, criterion)
      epoch_test_loss += loss.item()
    epoch_test_loss /= len(test_dataloader) 
    test_loss.append(epoch_test_loss)

    print(f"Epoch {epoch} of {n_epoch}: Training loss: {epoch_train_loss}, Test loss: {epoch_test_loss}")

  return train_loss, test_loss

##curve plotting
def plot_curve(train_loss, test_loss, train_curve_path):
  epochs = np.arange(len(train_loss))

  plt.figure()
  plt.plot(epochs, train_loss, 'b', label='training loss')
  plt.plot(epochs, test_loss, 'r', label='test loss')
  plt.title('Training and Test Loss Curve over Epochs')
  plt.xlabel('Epochs')
  plt.ylabel('L1loss')
  plt.legend()
  plt.savefig(train_curve_path) #saving path

##visualization
def load_img(img_path, model_input_size, device):
  img = Image.open(img_path).convert('RGB')
  # original_size = img.size
  ##preprocess image
  normalize = transforms.Normalize(
            mean=[0.458, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
  )
  img = img.resize((model_input_size, model_input_size))
  img = img_display = np.asarray(img)/255.0
  img = torch.tensor(img).permute(2, 0, 1)
  img = normalize(img)
  # Convert images and keypoints to the correct data type
  # img = img.float().to(device)  # Convert images to float32 and move to the correct device
         
  return img.to(device), img_display

def visualization(img_path, model, viz_result_path, model_input_size, device):
  # img_index = 10
  # img = test_data.load_img(img_index)
  img_tensor, img_display =load_img(img_path, model_input_size, device)

  plt.figure(figsize=(10,10))
  plt.subplot(121)
  plt.title('original image')

  plt.imshow(img_display)

  plt.subplot(122)
  plt.title('image with facial keypoints')
  plt.imshow(img_display)
  # img, _ = test_data[img_index]
  kp_s = model(img_tensor[None]).flatten().detach().cpu()
  plt.scatter(kp_s[:68]*model_input_size, kp_s[68:]*model_input_size, c='r',s=6, alpha=0.6, edgecolors='black', cmap='viridis')

  plt.savefig(viz_result_path)

      
