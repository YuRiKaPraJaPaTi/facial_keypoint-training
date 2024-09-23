# #inbuilt packages
# from PIL import Image

# #DataScience packages
# import numpy as np
# import matplotlib.pyplot as plt

# #Pytorch related packages
# import torch
# from torchvision import transforms

# class FacialKeyPointDetection:
#   def __init__(self) -> None:
        
#     self.model = torch.load('dump.version1.model.pth')
#     self.normalize = transforms.Normalize(
#       mean=[0.458, 0.456, 0.406],
#       std=[0.229, 0.224, 0.225]
#     )
#     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#   def predict(self, image):
#     img, img_display = self.preprocess(image)
#     kps = self.model(img[None]).flatten().detach().cpu()
#     kp_x, kp_y = self.postprocess(img_display, kps)
#     return img_display, (kp_x, kp_y)
  
#   def preprocess(self, img):
#     ##preprocess image
#     img = img.resize((224, 224))
#     img = img_display = np.asarray(img)/255.0
#     img = torch.tensor(img).permute(2, 0, 1)
#     img = self.normalize(img)    
#     # Convert images and keypoints to the correct data type
#     img = img.float()  # Convert images to float32 

#     return img.to(self.device), img_display
  
#   def postprocess(self, image, kps):
#     width, height, _ = image.shape
#     kp_x, kp_y = kps[:68]*width, kps[:68]*height
#     return kp_x, kp_y
  

import numpy as np

import torch 
from torchvision import transforms


class FacialKeyPointDetection:
    def __init__(self) -> None:
        self.model = torch.load('dump/version_2/model.pth')
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self, image):
        img, img_disp = self.preprocess(image)
        kps = self.model(img[None]).flatten().detach().cpu()
        kp_x, kp_y = self.postprocess(img_disp, kps)
        return img_disp, (kp_x, kp_y)
    
    def preprocess(self, img):
        img = img.resize((224, 224))
        img = img_disp = np.asarray(img)/ 225
        img = torch.tensor(img).permute(2, 0, 1).float()
        img = self.normalize(img).float()
        return img.to(self.device), img_disp

    def postprocess(self, img, kps):
        img = np.array(img)
        width, height, _ = img.shape
        kp_x, kp_y = kps[:68] * width, kps[68:] * height
        return kp_x, kp_y
