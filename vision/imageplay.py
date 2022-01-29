# Import the necessary libraries
from PIL import Image
import numpy as np
  
# load the image and convert into
# numpy array
img = Image.open('path.jpg')
  
# asarray() class is used to convert
# PIL images into NumPy arrays
numpydata = np.asarray(img)
print(numpydata.shape)