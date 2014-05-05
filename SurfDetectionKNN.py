import cv2
import numpy as np
import time
import gc
import cv
from matplotlib import pyplot as plt 
from scipy.cluster.vq import kmeans,vq
gc.disable()


class SURF():
	
	def __init__(self, Hessian_Threshold):

		self.Hessian_Threshold = Hessian_Threshold
	
	def getFeatures(self, imgg):
		surf = cv2.SURF(self.Hessian_Threshold)
		kp, descriptors = surf.detectAndCompute(imgg,None)
		return kp, descriptors

class SIFT():
	
	def getFeatures(self, img):
		sift = cv2.SIFT()
		return sift.detectAndCompute(img, None)


class Detector():
	
	#def __init__(self, Hessian_Threshold, dist_threshold):
	def __init__(self, dist_threshold):
		self.dist_threshold = dist_threshold
		#self.Hessian_Threshold = Hessian_Threshold
		#self.Feature_Extractor = SURF(self.Hessian_Threshold)
		self.Feature_Extractor = SIFT()
		self.knn = cv2.KNearest()
		self.keypoints = []
		self.samples = np.array([0 for i in range(128)], dtype = np.float32)
		self.height = None
		self.width  = None
		self.Number = 0

	def train(self, no):
# Load the images
		#import pdb;pdb.set_trace()
		imgg = cv2.imread('../Template1/' + '0001' + '.jpg')
		self.height = imgg.shape[0] 
		self.width = imgg.shape[1]  
		for i in range(1,no+1):
			seq = '%04d'%i
			imgg = cv2.imread('../Template1/' + str(seq) + '.jpg')
			#imgg =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			#import pdb;pdb.set_trace()
			kp, descriptors = self.Feature_Extractor.getFeatures(imgg)
			if descriptors != None:
				self.samples = np.vstack((self.samples,descriptors))
				self.keypoints = self.keypoints + kp
		self.samples = self.samples[1:]
		#responses = np.arange(self.samples.shape[0],dtype = np.float32)
		#self.knn.train(self.samples,responses)

# Now loading a template image and searching for similar keypoints
	def Detect(self, img):
		img1 = img.copy()
		
		keys,desc = self.Feature_Extractor.getFeatures(img)
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(desc,self.samples,k=2)
		good = []
		for m,n in matches:
			if m.distance < 0.7*n.distance:
				good.append(m)
		if len(good)>10:
			src_pts = np.float32([ keys[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ self.keypoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		import pdb;pdb.set_trace()
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()
		pts = np.float32([ [0,0],[0,self.height-1],[self.width-1,self.height-1],[self.width-1,0] ]).reshape(-1,1,2)

		'''
		retval, results, neigh_resp, dists = self.knn.find_nearest(desc,1)
		X, Y = [], []
		goodSrc = []
		goodDest = []
		color = (0,0,255)
		centers = np.array([0 for i in range(2)], dtype = np.float32)
		center = []
		for h in range(desc.shape[0]):
			des = desc[h].reshape((1,128))
			retval, results, neigh_resp, dists = self.knn.find_nearest(des,1)
			
			dist = dists[0][0]
			
			if dist < self.dist_threshold:
		#Draw matched key points on template image
				center.append((int(keys[h].pt[0]),int(keys[h].pt[1])))
				#import pdb;pdb.set_trace()
				centers  = np.vstack((centers, [int(keys[h].pt[0]),int(keys[h].pt[1])]))
				
				#cv2.circle(img,center,8,color,-1)
				goodSrc.append(keys[h].pt)
				goodDest.append(self.keypoints[int(neigh_resp)].pt)
		centers = centers[1:]
		centroids,_ = kmeans(centers,6)
		idx,_ = vq(centers,centroids)
		goodSrc1 = []
		goodDest1 = []
		'''
		VAL = self.maxFrequency(idx)
		import pdb;pdb.set_trace()
		for JJ in range(len(idx)):
			if idx[JJ] == VAL:
				goodSrc1.append(goodSrc[JJ])
				goodDest1.append(goodDest[JJ])
				cv2.circle(img1,center[JJ],8,color,-1)
		
		#import pdb;pdb.set_trace()
		#cv2.imwrite('{0:04d}.jpg'.format(self.Number),img)
		import pdb;pdb.set_trace()
		if len(goodSrc1) >= 4:
			src_pts = np.float32(goodSrc1).reshape(-1,1,2)
			dst_pts = np.float32(goodDest1).reshape(-1,1,2)
			M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5)
			
			matchesMask = mask.ravel().tolist()
		#import pdb;pdb.set_trace()
		
			pts = np.float32([ [0,0],[0,self.height-1],[self.width-1,self.height-1],[self.width-1,0] ]).reshape(-1,1,2)
			dst = cv2.perspectiveTransform(pts,M)
		
			cv2.polylines(img1,[np.int32(dst)],True,(0,255,255))
			
			cv2.imwrite('{0:04d}.jpg'.format(self.Number),img1)
			self.Number += 1
		
				
			#x, y = keys[h].pt
			#center = (int(x), int(y))
		# Calculate Homography here
		#center = (int(sum(X)/len(X)),int(sum(Y)/len(Y)))
		#return center
			CO_ORDS = [[dst[0][0][0], dst[0][0][1]],[dst[3][0][0], dst[3][0][1]],[dst[2][0][0], dst[2][0][1]],[dst[1][0][0], dst[1][0][1]]]
			#import pdb;pdb.set_trace()
			# Retraining this updates the appearance model
			if CO_ORDS[0][1] < CO_ORDS[2][1] and CO_ORDS[0][1] >= 0 and CO_ORDS[2][1] >=0:
				if CO_ORDS[0][0] < CO_ORDS[1][0] and CO_ORDS[0][0] >= 0 and CO_ORDS[1][0] >= 0:
					temp = img[CO_ORDS[0][1]:CO_ORDS[2][1], CO_ORDS[0][0]:CO_ORDS[1][0]]
					kp, descriptors = self.Feature_Extractor.getFeatures(temp)
					self.samples = np.vstack((self.samples,descriptors))
					self.keypoints = self.keypoints + kp
					responses = np.arange(self.samples.shape[0],dtype = np.float32)
					print 'Re Training'
					self.knn.train(self.samples,responses)
					return CO_ORDS
				else:return False
			else:return False
		else:
			return False
	
	def maxFrequency(self, A):
		dict = {}
		for i in A:
			if i not in dict:
				dict[i] = 1
			else:
				dict[i] = dict[i] + 1
		keys = dict.keys()
		max = 0
		val = 0
		for i in keys:
			if dict[i] > max:
				max = dict[i]
				val = i
		return val
			
