# IMPLEMENTING MEAN SHIFT IMAGE SEGMENTATION IN PYTHON
# Author(s): Pranshu Gupta, Abhishek Jain
###################################################################################################

import numpy as np
import math

###################################################################################################

def gaussian_mean(kernel,seed):
	# Initializing the mean
	num = np.matrix([0.0,0.0,0.0,0.0,0.0])
	den = 1.0

	# The positive definite matrix
	A = np.matrix([[1.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,1.0]])

	# Finding the mean
	for point in kernel:
		x = np.matrix([point[0]/rows,point[1]/cols,point[2]/255.0,point[3]/255.0,point[4]/255.0])
		g = (-1.0)*x*A*(x.transpose())
		fx = math.exp(g)
		num = num + x*fx
		den = den + fx

	mean = num/den
	mean = np.array(mean).flatten()
	mean = np.array([mean[0]*rows,mean[1]*cols,mean[2]*255.0,mean[3]*255.0,mean[4]*255.0], dtype=np.int64)

	return mean