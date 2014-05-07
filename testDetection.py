import numpy as np
import cv2


MIN_MATCH_COUNT = 10

img1 = cv2.imread('../Template1/' + '0001' + '.jpg',0) # queryImage
img2 = cv2.imread('/home/ankush/OriginalNN/NNTracker/src/Data/nl_cereal_s5/frame00167.jpg',0) # trainImage

sift = cv2.SIFT()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 25)
search_params = dict(checks = 10)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []

for m,n in matches:
	if m.distance < 0.9*n.distance:
		good.append(m)

if len(good) > MIN_MATCH_COUNT:
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
	
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	matchesMask = mask.ravel().tolist()
	
	h,w = img1.shape
	#import pdb;pdb.set_trace()
	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	
	dst = cv2.perspectiveTransform(pts,M)
	cv2.polylines(img2,[np.int32(dst)],True,(0,255,255))
else:
	print 'Not enough key points'
	matchesMask = None
cv2.imwrite('hjk.jpg',img2)
print 'Done'


'''
M = 
array([[  8.77885247e-01,   1.74892943e-04,  -2.19279400e+02],
       [ -3.12511700e-02,   8.82958924e-01,  -1.30205084e+02],
       [ -2.19339459e-04,  -5.00991102e-05,   1.00000000e+00]])
'''
