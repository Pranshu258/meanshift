# IMPLEMENTING MEAN SHIFT IMAGE SEGMENTATION IN PYTHON
# Author(s): Pranshu Gupta, Abhishek Jain
###################################################################################################

import numpy as np
import math

###################################################################################################

def gaussian_mean(kernel,seed,bandwidth):
	weights = np.exp(-1*np.linalg.norm((kernel - seed)/bandwidth,axis=1))
	mean = np.array(np.sum(weights[:,None]*kernel,axis=0)/np.sum(weights), dtype=np.int64)
	return mean