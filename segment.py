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
Bin = None
kertype = "flat"

if len(sys.argv) == 4:
	bandwidth = int(sys.argv[1])
	Bin = int(sys.argv[2])
	gaussian = int(sys.argv[3])
else:
	print "Usage: python segment.py bandwidth seeding_bin_size gaussian"
	exit()

if gaussian == 1:
	kertype = "gaussian"

m = 1
S = 5
threshold = 1.0

img = Image.open("img/input.jpg")
img.load()
img = np.array(img)

seg_img = img

rows, cols, dim = img.shape

seg_meandist = np.array([[1000.0 for r in xrange(cols)] for c in xrange(rows)])
meandist = np.array([[1000.0 for r in xrange(cols)] for c in xrange(rows)])

# Get the kernel for a point
start = t.time()
means = []
for r in xrange(0,rows,Bin):
	for c in xrange(0,cols,Bin):
		seed = np.array([r,c,img[r][c][0],img[r][c][1],img[r][c][2]])
		for n in xrange(10):
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
						meandist[i][j] = D
			kernel = np.array(kernel)
			# Get the mean of the kernel which will be used as the new seed
			
			# Kernel
			if gaussian == 0:
				mean = np.mean(kernel,axis=0,dtype=np.int64)
			elif gaussian == 1:
				mean = gaussian_mean(kernel, float(rows), float(cols))

			# Get the shift
			dc = np.linalg.norm(seed[2:] - mean[2:])
			ds = (np.linalg.norm(mean[:2] - mean[:2]))*m/S
			dsm = np.linalg.norm([dc,ds])
			seed = mean
			if dsm <= threshold:
				break
		means.append(seed)

		# Check if this seed is better than the last assigned color for the pixels in kernel
		for k in kernel:
			i = k[0]
			j = k[1]
			D = np.linalg.norm([meandist[i][j],dsm])
			if D < seg_meandist[i][j]:
				seg_meandist[i][j] = D
				seg_img[i][j] = mean[2:]

end = t.time()

img = Image.fromarray(seg_img)
img.save("img/" + kertype + "_output_" + str(bandwidth) + "_" + str(Bin) + ".jpg")

print "Time taken for segmentation: " + str((end - start)/60) + " min"
print "Bandwidth: " + str(bandwidth)
print "Seeds Bin Size: " + str(Bin)
print "Number of Seeds: " + str(len(means)) 
print "Relative weight given to spatial distance (m): " + str(m)
print "Spatial Distance Normalization Parameter (S): " + str(S)