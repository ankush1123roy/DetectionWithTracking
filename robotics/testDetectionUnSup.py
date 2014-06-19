import numpy as np
import cv2
from processFilterBanks import *
import pyflann
from numpy import matrix as MA

def indexCalculator(N, I):
	return (I/(N), I%(N))

MIN_MATCH_COUNT = 10

img1 = cv2.imread('../Template1/' + '0001' + '.jpg') # queryImage
img2 = cv2.imread('/home/ankush/OriginalNN/NNTracker/src/Data/Liquor/img/0001.jpg') # trainImage
import pdb;pdb.set_trace()
QH, QW, _  = img1.shape
TH, TW, _  = img2.shape



des1 = np.float32(processFilterBanks(img1))
des2 = np.float32(processFilterBanks(img2))
color = (112,0,112)

'''
flann = pyflann.FLANN()
flann.build_index(des1, algorithm='kdtree', trees=10)
search_params = dict(checks = 50)

import pdb;pdb.set_trace()
for i in range(len(des2)):
	results, dists = flann.nn_index(des2[i])
	if dists == 0:
		KK = indexCalculator(TW, i)
		print KK
		cv2.circle(img2,KK,4,color,-1)
cv2.imwrite('ksgj.jpg',img2)
'''
'''
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 25)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)
import pdb;pdb.set_trace()
for m,n in matches:
	if m.distance < 0.2*n.distance:
		KK = indexCalculator(TW, m.trainIdx)
		print KK
		cv2.circle(img2,KK,4,color,-1)
cv2.imwrite('ksgj.jpg',img2)
'''
knn = cv2.KNearest()
responses = np.arange(len(des1), dtype = np.float32)
knn.train(MA(des1), responses)
for i in range(len(des2)):
	ind, _, neigh, dist = knn.find_nearest(MA(des2[i]), 2)
	if dist[0][0] == 0:
		cv2.circle(img2,indexCalculator(TW, i),4,color,-1)
cv2.imwrite('ksgj.jpg',img2)

'''
# cv2.circle(img,center,8,color,-1)
I = 0.1
J  = 0 
while I < 1:
	good = []
	for m,n in matches:
		if m.distance < I*n.distance:
			good.append(m)
#import pdb; pdb.set_trace()

	if len(good) > MIN_MATCH_COUNT:
	
		src_pts = np.float32([ indexCalculator(QW, m.queryIdx) for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ indexCalculator(TW, m.trainIdx) for m in good ]).reshape(-1,1,2)
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()
	
	#h,w = img1.shape
	#import pdb;pdb.set_trace()
	#pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		pts = np.float32([ [0,0],[0,QH-1],[QW-1,QH-1],[QW-1,0] ]).reshape(-1,1,2)
	
		dst = cv2.perspectiveTransform(pts,M)
		cv2.polylines(img2,[np.int32(dst)],True,(0,255,0))
	else:
		print 'Not enough key points'
		matchesMask = None
	cv2.imwrite('{0:04d}.jpg'.format(J),img2)
	print 'Done'
	I = I + 0.1
	J += 1
'''

	
	
	
