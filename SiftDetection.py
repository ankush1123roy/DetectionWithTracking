import numpy as np
import cv2


class SIFT():
	
	def getFeatures(self, img):
		sift = cv2.SIFT()
		return sift.detectAndCompute(img, None)


class Detector():
	
	def __init__(self, ratio_threshold, MIN_MATCH_COUNT):
		
		self.ratio_threshold = ratio_threshold
		self.Feature_Extractor = SIFT()
		self.knn = cv2.KNearest()
		self.keypoints = []
		self.samples =  np.array([0 for i in range(128)], dtype = np.float32)
		self.height = None
		self.width  = None
		self.Feature_Extractor = SIFT()
		self.Number = 0
		self.minMatch = MIN_MATCH_COUNT
		self.Number = 0
	
	def train(self):
		img1 = cv2.imread('../Template1/' + '0002' + '.jpg',0) # queryImage
#img2 = cv2.imread('/home/ankush/OriginalNN/NNTracker/src/Data/Liquor/img/1467.jpg',0) # trainImage
		
		self.height, self.width = img1.shape 
		#import pdb; pdb.set_trace()
		#self.keypoints = kp2
		#self.samples =  des2
		
		for i in range(5,6):
			seq = '%04d'%i
			print 'Training on', str(seq), '.jpg'
			img1 = cv2.imread('../Template1/' + str(seq) + '.jpg')
			kp2, des2 = self.Feature_Extractor.getFeatures(img1)
			#import pdb;pdb.set_trace()
			if len(des2) > 0:
				self.samples = np.vstack((self.samples,des2))
				self.keypoints = self.keypoints + kp2
		#import pdb;pdb.set_trace()
		self.samples = self.samples[1:]
		#self.keypoints = self.keypoints[1:]



	def Detect(self, img):
		img1 = img.copy()
		#import pdb;pdb.set_trace()
		kp1, des1 = self.Feature_Extractor.getFeatures(img)
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
		search_params = dict(checks = 50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(des1, self.samples, k=2)
		good = []
		for m,n in matches:
			if m.distance < self.ratio_threshold*n.distance:
				good.append(m)

		if len(good) > self.minMatch:
			dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			src_pts = np.float32([ self.keypoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
			matchesMask = mask.ravel().tolist()

			pts = np.float32([ [0,0],[0,self.height-1],[self.width-1,self.height-1],[self.width-1,0] ]).reshape(-1,1,2)
			#import pdb ;pdb.set_trace()
			dst = cv2.perspectiveTransform(pts,M)
			cv2.polylines(img1,[np.int32(dst)],True,(0,255,255))
			cv2.imwrite('{0:04d}.jpg'.format(self.Number), img1)
			#cv2.imwrite('hjk.jpg',img1)
			self.Number = self.Number + 1
			CO_ORDS = [[dst[0][0][0], dst[0][0][1]],[dst[3][0][0], dst[3][0][1]],[dst[2][0][0], dst[2][0][1]],[dst[1][0][0], dst[1][0][1]]]
			if CO_ORDS[0][1] < CO_ORDS[2][1] and CO_ORDS[0][1] >= 0 and CO_ORDS[2][1] >=0:
				if CO_ORDS[0][0] < CO_ORDS[1][0] and CO_ORDS[0][0] >= 0 and CO_ORDS[1][0] >= 0:
					temp = img[CO_ORDS[0][1]:CO_ORDS[2][1], CO_ORDS[0][0]:CO_ORDS[1][0]]
					kp, descriptors = self.Feature_Extractor.getFeatures(temp)
					self.samples = np.vstack((self.samples,descriptors))
					self.keypoints = self.keypoints + kp
			return CO_ORDS
		else:
			print 'Not enough key points'
			matchesMask = None
			return False
'''
M = 
array([[  9.97853213e-01,   2.45424613e-02,   2.49750175e+02],
       [ -5.70531669e-02,   1.06755581e+00,   1.58148134e+02],
       [ -1.79983110e-04,   8.76030140e-05,   1.00000000e+00]])
'''
