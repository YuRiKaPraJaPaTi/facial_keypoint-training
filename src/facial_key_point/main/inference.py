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
  image=Image.open('face.jpg').convert('RGB')
  facial_key_point_detection = FacialKeyPointDetection()
  image, kp = facial_key_point_detection.predict(image)
  print(image)

  plt.figure()
  plt.imshow(image)
  plt.scatter(kp[0], kp[1], s=5, c='r')
  plt.savefig('viz.png')


# from PIL import Image
# from matplotlib import pyplot as plt

# from src.facial_key_point.utils.inference_help import FacialKeyPointDetection

# if __name__ == "__main__":
#     image = Image.open('face.jpg').convert('RGB')
#     facial_key_point_detection = FacialKeyPointDetection()
#     image, kp = facial_key_point_detection.predict(image)

#     plt.figure()
#     plt.imshow(image)
#     plt.scatter(kp[0], kp[1], s=4, c='r')
#     plt.savefig('viz.png')