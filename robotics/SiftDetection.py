import numpy as np
import cv2
from scipy.cluster.vq import kmeans,vq


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
		img1 = cv2.imread('../../Template1/' + '0001' + '.jpg',0) # queryImage
		
		self.height, self.width = img1.shape 
		for i in range(1,3):
			seq = '%04d'%i
			print 'Training on', str(seq), '.jpg'
			img1 = cv2.imread('../../Template1/' + str(seq) + '.jpg')
			kp2, des2 = self.Feature_Extractor.getFeatures(img1)
			if len(des2) > 0:
				self.samples = np.vstack((self.samples,des2))
				self.keypoints = self.keypoints + kp2
		self.samples = self.samples[1:]



	def Detect(self, img):
		print 'Detecting The Object'
		img1 = img.copy()
		#import pdb;pdb.set_trace()
		kp1, des1 = self.Feature_Extractor.getFeatures(img)
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
		search_params = dict(checks = 20)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(des1, self.samples, k=2)
		good = []
		color = (0,0,255)
		for m,n in matches:
			if m.distance < self.ratio_threshold*n.distance:
				good.append(m)

		if len(good) > self.minMatch:
			dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
			src_pts = np.float32([ self.keypoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
			tmp = []
			for j in range(dst_pts.shape[0]):
				tmp.append([dst_pts[j][0][0],dst_pts[j][0][1]])
			tmp = np.array(tmp)
			#import pdb;pdb.set_trace()
			centroids,_ = kmeans(tmp,4)
			idx,_ = vq(tmp,centroids)
			major = {}
			for l in idx:
				if l not in major: 
					major[l] = 1
				else:
					major[l] = major[l] + 1
			items = major.items()
			items.sort(key=lambda x: x[1])
			#import pdb;pdb.set_trace()
			major = items[-1][0]
			goodDst = []
			goodSrc = []
			for k in range(len(idx)):
				if idx[k] == major:
					goodDst.append([dst_pts[k][0][0], dst_pts[k][0][1]])
					goodSrc.append([src_pts[k][0][0], src_pts[k][0][1]])
			goodDst = np.float32(goodDst)
			goodSrc = np.float32(goodSrc)
			#import pdb;pdb.set_trace()
			for i in range(len(goodDst)):
				cv2.circle(img1,(goodDst[i][0], goodDst[i][1]),8,color,-1)
			M, mask = cv2.findHomography(goodSrc, goodDst, cv2.RANSAC,2.0)
			matchesMask = mask.ravel().tolist()

			pts = np.float32([[0,0],[0,self.height-1],[self.width-1,self.height-1],[self.width-1,0] ]).reshape(-1,1,2)
			dst = cv2.perspectiveTransform(pts,M)
			cv2.polylines(img1,[np.int32(dst)],True,(0,255,255))
			cv2.imwrite('{0:04d}.jpg'.format(self.Number), img1)
			self.Number = self.Number + 1
			CO_ORDS = [[dst[0][0][0], dst[0][0][1]],[dst[3][0][0], dst[3][0][1]],[dst[2][0][0], dst[2][0][1]],[dst[1][0][0], dst[1][0][1]]]
			
			if CO_ORDS[0][1] < CO_ORDS[2][1] and CO_ORDS[0][1] >= 0 and CO_ORDS[2][1] >=0:
				if CO_ORDS[0][0] < CO_ORDS[1][0] and CO_ORDS[0][0] >= 0 and CO_ORDS[1][0] >= 0:
					temp = img[CO_ORDS[0][1]:CO_ORDS[2][1], CO_ORDS[0][0]:CO_ORDS[1][0]]
					#import pdb;pdb.set_trace()
					kp, descriptors = self.Feature_Extractor.getFeatures(temp)
					self.samples = np.vstack((self.samples,descriptors))
					self.keypoints = self.keypoints + kp
			
			return CO_ORDS

			print 'Not enough key points'
			matchesMask = None
			return False

