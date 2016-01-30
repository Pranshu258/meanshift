# IMPLEMENTING MEAN SHIFT IMAGE SEGMENTATION IN PYTHON
# Author(s): Pranshu Gupta, Abhishek Jain
###################################################################################################

import Image
import numpy as np
import time as t
import sys
from gaussian_mean import gaussian_mean

###################################################################################################
bandwidth = None
Bin = 40
kertype = "flat"


if len(sys.argv) == 3:
	bandwidth = int(sys.argv[1])
	gaussian = int(sys.argv[2])
else:
	# print "Usage: python segment.py bandwidth do_gaussian"
	exit()

if gaussian == 1:
	kertype = "gaussian"

m = 1
S = 5
threshold = 1.0

# print "Loading the Image ..."
img = Image.open("img/input.jpg")
img.load()
img = np.array(img)

seg_img = img

rows, cols, dim = img.shape

meandist = np.array([[1000.0 for r in xrange(cols)] for c in xrange(rows)])
labels = np.array([[-1 for r in xrange(cols)] for c in xrange(rows)])

# print "Running the Mean Shift algorithm ..."

start = t.time()

means = []
for r in xrange(0,rows,Bin):
	for c in xrange(0,cols,Bin):
		seed = np.array([r,c,img[r][c][0],img[r][c][1],img[r][c][2]])
		for n in xrange(15):
			x = seed[0]
			y = seed[1]
			r1 = max(0,x-bandwidth*5)
			r2 = min(r1+bandwidth*10, rows)
			c1 = max(0,y-bandwidth*5)
			c2 = min(c1+bandwidth*10, cols)
			kernel = []
			for i in xrange(r1,r2):
				for j in xrange(c1,c2):
					dc = np.linalg.norm(img[i][j] - seed[2:])
					ds = (np.linalg.norm(np.array([i,j]) - seed[:2]))*m/S
					D = np.linalg.norm([dc,ds])
					if D < bandwidth:
						kernel.append([i,j,img[i][j][0],img[i][j][1],img[i][j][2]])
			kernel = np.array(kernel)			

			if gaussian == 0:
				mean = np.mean(kernel,axis=0,dtype=np.int64)
			elif gaussian == 1:
				mean = gaussian_mean(kernel, seed, bandwidth)

			# Get the shift
			dc = np.linalg.norm(seed[2:] - mean[2:])
			ds = (np.linalg.norm(seed[:2] - mean[:2]))*m/S
			dsm = np.linalg.norm([dc,ds])
			seed = mean
			if dsm <= threshold:
				# print "Mean " + str(len(means)+1) + " converges in: " + str(n) + " iterations"
				break
		means.append(seed)

end = t.time()

# print "Grouping together the means that are closer than the bandwidth ..."
flags = [1 for me in means]
for i in xrange(len(means)):
	if flags[i] == 1:
		w = 1.0
		j = i + 1
		while j < len(means):
			dc = np.linalg.norm(means[i][2:] - means[j][2:])
			ds = (np.linalg.norm(means[i][:2] - means[j][:2]))*m/S
			dsm = np.linalg.norm([dc,ds])
			if dsm < bandwidth:
				means[i] = means[i] + means[j]
				w = w + 1.0
				flags[j] = 0
			j = j + 1
		means[i] = means[i]/w
converged_means = []
for i in xrange(len(means)):
	if flags[i] == 1:
		converged_means.append(means[i])
converged_means = np.array(converged_means)

# # print "Constructing the segmented image ..."
# for i in xrange(rows):
# 	for j in xrange(cols):
# 		for c in xrange(len(converged_means)):
# 			dc = np.linalg.norm(img[i][j] - converged_means[c][2:])
# 			ds = (np.linalg.norm(np.array([i,j]) - converged_means[c][:2]))*m/S
# 			D = np.linalg.norm([dc,ds])
# 			if D < meandist[i][j]:
# 				meandist[i][j] = D
# 				labels[i][j] = c
# 		seg_img[i][j] = converged_means[labels[i][j]][2:] 

# print "Saving the segmented image ..."
# seg_img = Image.fromarray(seg_img)
# seg_img.save("img/" + kertype + "_output_" + str(bandwidth) + ".jpg")

# print "Time taken for segmentation: " + str((end - start)/60) + " min"
# print "Bandwidth: " + str(bandwidth)
# print "Number of Means before convergence: " + str(len(means))
# print "Number of Means after convergence: " + str(len(converged_means))

print bandwidth, len(converged_means)