import numpy as np
import cv2
from scipy import signal
from numpy import matrix as MA
import pyflann
from sklearn.preprocessing import scale

def processFilterBanks(img):
	
	#img = cv2.imread('/home/ankush/OriginalNN/NNTracker/src/Data/Liquor/img/0001.jpg')
	imgg =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	FILTERS = np.array((open("Liquor.txt",'r').readlines()))
	# output.txt has the filter banks
	response = []

	for i in range(len(FILTERS)):
		FILTER = [float(j) for j in FILTERS[i].split()]
		FILTER = np.reshape(MA(FILTER),(5,5)).T
		#FeatureMap = signal.convolve2d(imgg, FILTER,mode='full', boundary='fill', fillvalue=0)
		FeatureMap = signal.convolve(imgg, FILTER, "valid")
		#import pdb;pdb.set_trace()
		response.append(FeatureMap.flatten())
	#FEATURES = generateFeatureVectors(response)
	return generateFeatureVectors(response)

def generateFeatureVectors(responses):
	FEATURE_MATRIX = []
	for i in range(len(responses[0])):
		temp = []
		for j in range(len(responses)):
			temp.append(responses[j][i])
		FEATURE_MATRIX.append(scale( temp, axis=0, with_mean=True, with_std=True, copy=True ))
	return np.array(FEATURE_MATRIX)


if __name__ == '__main__':
	processFilterBanks()
