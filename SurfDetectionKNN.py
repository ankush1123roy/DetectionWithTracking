import cv2
import numpy as np
import time
import gc
from matplotlib import pyplot as plt 
gc.disable()


class SURF():
	
	def __init__(self, Hessian_Threshold):

		self.Hessian_Threshold = Hessian_Threshold
	
	def getFeatures(self, imgg):
		surf = cv2.SURF(self.Hessian_Threshold)
		kp, descriptors = surf.detectAndCompute(imgg,None)
		return kp, descriptors


class Detector():
	
	def __init__(self, Hessian_Threshold, dist_threshold):
		self.dist_threshold = dist_threshold
		self.Hessian_Threshold = Hessian_Threshold
		self.Feature_Extractor = SURF(self.Hessian_Threshold)
		self.knn = cv2.KNearest()
		self.keypoints = []
		self.samples = np.array([0 for i in range(128)], dtype = np.float32)
		self.height = None
		self.width  = None
		self.Number = 0

	def train(self, no):
# Load the images
		#import pdb;pdb.set_trace()
		imgg = cv2.imread('../Template/' + '0001' + '.jpg')
		self.height = imgg.shape[0] 
		self.width = imgg.shape[1]  
		for i in range(1,no+1):
			seq = '%04d'%i
			imgg = cv2.imread('../Template/' + str(seq) + '.jpg')
			#imgg =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			#import pdb;pdb.set_trace()
			kp, descriptors = self.Feature_Extractor.getFeatures(imgg)
			self.samples = np.vstack((self.samples,descriptors))
			self.keypoints = self.keypoints + kp
		self.samples = self.samples[1:]
		responses = np.arange(self.samples.shape[0],dtype = np.float32)
		self.knn.train(self.samples,responses)

# Now loading a template image and searching for similar keypoints
	def Detect(self, img):
		#import pdb;pdb.set_trace()
		keys,desc = self.Feature_Extractor.getFeatures(img)
		retval, results, neigh_resp, dists = self.knn.find_nearest(desc,1)
		X, Y = [], []
		goodSrc = []
		goodDest = []
		for h in range(desc.shape[0]):
			des = desc[h].reshape((1,128))
			retval, results, neigh_resp, dists = self.knn.find_nearest(des,1)
			
			dist = dists[0][0]
			
			if dist < self.dist_threshold:
		#Draw matched key points on template image
				X.append(keys[h].pt[0]);Y.append(keys[h].pt[1])
				goodSrc.append(keys[h].pt)
				goodDest.append(self.keypoints[int(neigh_resp)].pt)
		#import pdb;pdb.set_trace()
		src_pts = np.float32(goodSrc).reshape(-1,1,2)
		dst_pts = np.float32(goodDest).reshape(-1,1,2)
		M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,2.0)
		matchesMask = mask.ravel().tolist()
		#import pdb;pdb.set_trace()
		
		pts = np.float32([ [0,0],[0,self.height-1],[self.width-1,self.height-1],[self.width-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)
		
		cv2.polylines(img,[np.int32(dst)],True,(0,255,255))
		cv2.imwrite('{0:04d}.jpg'.format(self.Number),img)
		self.Number += 1
		
				
			#x, y = keys[h].pt
			#center = (int(x), int(y))
		# Calculate Homography here
		#center = (int(sum(X)/len(X)),int(sum(Y)/len(Y)))
		#return center
		CO_ORDS = [[dst[0][0][0], dst[0][0][1]],[dst[3][0][0], dst[3][0][1]],[dst[2][0][0], dst[2][0][1]],[dst[1][0][0], dst[1][0][1]]]
		return CO_ORDS
