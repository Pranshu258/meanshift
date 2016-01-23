# IMPLEMENTING MEAN SHIFT IMAGE SEGMENTATION IN PYTHON
# Author(s): Pranshu Gupta, Abhishek Jain
###################################################################################################

import Image
import numpy as np

###################################################################################################
# CONSTRUCT THE IMAGE DATA MATRIX

# Load the image
print "Loading the Image ..."
img = Image.open("img1.jpg")
img.load()
img_data = np.array(img)

# Height and Width of the Image
height = float(len(img_data))
width = float(len(img_data[0]))

# Constructing the NORMALIZED matrix of the image
print "Constructing the mormalized matrix of the Image ..."
i = 0
data = []
for row in img_data:
	j = 0
	for pixel in row:
		data.append([i, j, pixel[0], pixel[1], pixel[2]])
		j = j + 1
	i = i + 1

# Data is a numpy array of NORMALIZED [x,y,r,g,b] values of pixels in the image
# The dimension of this array with be h*w*5
data = np.array(data)

###################################################################################################
