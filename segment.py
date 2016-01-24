# IMPLEMENTING MEAN SHIFT IMAGE SEGMENTATION IN PYTHON
# Author(s): Pranshu Gupta, Abhishek Jain
###################################################################################################

import Image
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.cluster import estimate_bandwidth, get_bin_seeds
from sklearn.utils import extmath
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances


###################################################################################################
# CONSTRUCT THE IMAGE DATA MATRIX
def get_data(img_data):
	i = 0
	data = []
	for row in img_data:
		j = 0
		for pixel in row:
			data.append([i, j, pixel[0], pixel[1], pixel[2]])
			j = j + 1
		i = i + 1
	data = np.array(data)
	return data

###################################################################################################
# SELECT THE INITIAL KERNEL CENTERS
def get_seeds():

	return seeds

###################################################################################################

print "Loading the Image ..."
img = Image.open("img1.jpg")
img.load()
img = np.array(img)

img_height = len(img)					# Dimensions of the image
img_width = len(img[0])					#

print "Constructing the feature matrix ..."
data = get_data(img)

# print "Estimating the bandwidth of the Kernel ..."
# bandwidth = estimate_bandwidth(data, quantile=0.3, n_samples=None, random_state=0)
# print bandwidth

bandwidth = (img_height + img_width)/2.0
print "Getting the initial kernel seeds ..."
seeds = get_bin_seeds(data, bandwidth, 2)
print "Number of kernel seeds chosen: " + str(len(seeds))

