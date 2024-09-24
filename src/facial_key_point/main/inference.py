#inbuilt packages
from PIL import Image

#DataScience packages
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# import from files
from src.facial_key_point.utils.inference_help import FacialKeyPointDetection
  

if __name__=='__main__':
  #load and convert image
  image=Image.open('face3.jpg').convert('RGB')

  plt.figure(figsize=(10, 5))
  #plot original image
  plt.subplot(121)
  plt.title('Original Image')
  plt.imshow(image)

  #perform facial key point detection
  facial_key_point_detection = FacialKeyPointDetection()
  image, kp = facial_key_point_detection.predict(image)
  # print(image)

  #plot image with facial key point
  plt.subplot(122)
  plt.title("Image with Facial Keypoints")
  # plt.figure()
  plt.imshow(image)
  plt.scatter(kp[0], kp[1], s=5, c='r')
  plt.savefig('vis_face3.png')


