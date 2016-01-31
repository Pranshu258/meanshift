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


if len(sys.argv) == 4:
	filename = str(sys.argv[1])
	bandwidth = int(sys.argv[2])
	gaussian = int(sys.argv[3])
else:
	print "Usage: python segment.py bandwidth do_gaussian"
	exit()

if gaussian == 1:
	kertype = "gauss"


m = 1
S = 5
threshold = 1.0

print "Loading the Image " + filename + ".jpg"
img = Image.open("img/" + filename + ".jpg")
img.load()
img = np.array(img)

seg_img = img

rows, cols, dim = img.shape

meandist = np.array([[1000.0 for r in xrange(cols)] for c in xrange(rows)])
labels = np.array([[-1 for r in xrange(cols)] for c in xrange(rows)])

print "Running the Mean Shift algorithm ..."

start = t.time()

means = []
for r in xrange(0,rows,Bin):
	for c in xrange(0,cols,Bin):
		seed = np.array([r,c,img[r][c][0],img[r][c][1],img[r][c][2]])
		for n in xrange(15):
			x = seed[0]
			y = seed[1]
			r1 = max(0,x-Bin)
			r2 = min(r1+Bin*2, rows)
			c1 = max(0,y-Bin)
			c2 = min(c1+Bin*2, cols)
			kernel = []
			for i in xrange(r1,r2):
				for j in xrange(c1,c2):
					dc = np.linalg.norm(img[i][j] - seed[2:])
					ds = (np.linalg.norm(np.array([i,j]) - seed[:2]))*m/S
					D = np.linalg.norm([dc,ds])
					if D < bandwidth:
						kernel.append([i,j,img[i][j][0],img[i][j][1],img[i][j][2]])
			kernel = np.array(kernel)			
			# print kernel
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

print "Time taken for mean shift: " + str((end - start)/60) + " min"

print "Grouping together the means that are closer than the bandwidth ..."
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

# print "Number of Seeds: " + str(len(means))
# print "Number of Means: " + str(len(converged_means))

print "Constructing the segmented image ..."
for i in xrange(rows):
	for j in xrange(cols):
		for c in xrange(len(converged_means)):
			dc = np.linalg.norm(img[i][j] - converged_means[c][2:])
			ds = (np.linalg.norm(np.array([i,j]) - converged_means[c][:2]))*m/S
			D = np.linalg.norm([dc,ds])
			if D < meandist[i][j]:
				meandist[i][j] = D
				labels[i][j] = c
		seg_img[i][j] = converged_means[labels[i][j]][2:] 

print "Saving the segmented image ..."
seg_img = Image.fromarray(seg_img)
seg_img.save("img/" + kertype + "_output_" + filename + "_" + str(bandwidth) + ".jpg")

print bandwidth, len(converged_means)