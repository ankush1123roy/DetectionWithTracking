"""
Implementation of the Nearest Neighbour Tracking Algorithm.
Author: Travis Dick (travis.barry.dick@gmail.com)
"""
import time
import numpy as np
import pyflann
from scipy import weave
from scipy.weave import converters

from Homography import *
from ImageUtils import *
from SCVUtils import *
from TrackerBase import *
from SurfDetectionKNN import *
from numpy.linalg import inv

import cv2
import pdb
import sys
import random
import math

class NNTracker(TrackerBase):

	def __init__(self, n_samples, n_iterations=10, res=(20,20), warp_generator=lambda:random_homography(0.01, 0.01), use_scv=True):
		""" An implemetation of the Nearest Neighbour Tracker. 
	
				Parameters:
				-----------
				n_samples : integer
				The number of sample motions to generate. Higher values will improve tracking
				accuracy but increase running time.

				n_iterations : integer
				The number of times to update the tracker state per frame. Larger numbers
				may improve convergence but will increase running time.

				res : (integer, integer)
				The desired resolution of the template image. Higher values allow for more
				precise tracking but increase running time.

				warp_generator : () -> (3,3) numpy matrix.
				A function that randomly generates a homography. The distribution should
				roughly mimic the types of motions that you expect to observe in the 
				tracking sequence. random_homography seems to work well in most applications.

				See Also:
      
				---------
				TrackerBase
				BakerMatthewsICTracker
				"""
		self.n_samples = n_samples
		self.n_iterations = n_iterations
		self.res = res
		self.resx = res[0]
		self.resy = res[1]
		self.warp_generator = warp_generator
		self.n_points = np.prod(res)
		self.initialized = False
		self.pts = res_to_pts(self.res)
		self.use_scv=use_scv
		self.sift = False


 
	def set_region(self, corners):
		self.proposal = square_to_corners_warp(corners)
	
	def initialize(self, img, region):
		self.set_region(region)
		self.template = sample_and_normalize(img, self.pts, self.get_warp())
		FirstImage = np.reshape(self.template,(40,40))
		self.warp_index = _WarpIndex(self.n_samples, self.warp_generator, img, self.pts, self.get_warp(),self.res)
		self.intensity_map = None
		self.initialized = True

	def is_initialized(self):
		return self.initialized

	def update(self, img):
		W = True
		if not self.is_initialized(): return None
		for i in xrange(self.n_iterations):
			sampled_img = sample_and_normalize(img, self.pts, warp=self.proposal)
			if self.use_scv and self.intensity_map != None: sampled_img = scv_expectation(sampled_img, self.intensity_map)
				# --sift-- #
			if self.sift == False:
				self.proposal = self.proposal * self.warp_index.best_match(sampled_img)
				self.proposal /= self.proposal[2,2]
			else:
				temp_desc = self.pixel2sift(sampled_img)
		update = self.warp_index.best_match(sampled_img) 
		self.proposal = self.proposal * update
		self.proposal /= self.proposal[2,2]
		H =  self.warp_index.best_match(sampled_img)
		#sampled_img = np.reshape(sampled_img, (40,40))#, order = 'F')
		SSD = (sum([abs(sampled_img[k] - self.template[k]) for k in range(self.template.shape[0])])/self.template.shape[0])
		if SSD > 20:
			W = False
		#cv2.imwrite('../{0:04d}.jpg'.format(random.randint(1,1200)),sampled_img) # These are the sampled Images Can use them for training
		#out = cv2.warpPerspective(sampled_img, H, (640,480)) 
		#cv2.imwrite('../{0:04d}.jpg'.format(random.randint(1,1200)),np.reshape(sampled_img,(40,40))) 
		#out = np.array(out, dtype = 'int')
		
		#print self.proposal
#		cv2.imwrite('../{0:04d}.jpg'.format(random.randint(1,200)),out)
		if self.use_scv: self.intensity_map = scv_intensity_map(sample_region(img, self.pts, self.get_warp()), self.template)
		return self.proposal, W

	def get_warp(self):
		return self.proposal

	def get_region(self): # This is the part where he is getting the points
		return apply_to_pts(self.get_warp(), np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T)



class _WarpIndex:
	""" Utility class for building and querying the set of reference images/warps. """
	def __init__(self, n_samples, warp_generator, img, pts, initial_warp,res):
		self.resx = res[0]
		self.resy = res[1]
		self.sift = False
		self.indx = []
		n_points = pts.shape[1]
		print "Sampling Warps..."
		self.warps = [np.asmatrix(np.eye(3))] + [warp_generator() for i in xrange(n_samples - 1)]
		print "Sampling Images..."
		self.images = np.empty((n_points, n_samples))
		for i,w in enumerate(self.warps):
			self.images[:,i] = sample_and_normalize(img, pts, initial_warp * w.I)
		print "Building FLANN Index..."
		if self.sift == False:
				self.flann = pyflann.FLANN()
				self.flann.build_index(self.images.T, algorithm='kdtree', trees=10)
		else:
			desc = self.list2array(self.pixel2sift(self.images))
			self.flann = pyflann.FLANN()
		print "Done!"

    # --- For sift --- #
#	def pixel2sift(self,images):
#		detector = cv2.FeatureDetector_create("SIFT")
#	detector.setDouble('edgeThreshold',30)
#        descriptor = cv2.DescriptorExtractor_create("SIFT")
        #sift = cv2.SIFT(edgeThreshold = 20)
	# -- store descriptors in list --#
#        desc = []
#        for i in range(images.shape[1]):
#            patch = (images[:,i].reshape(self.resx,self.resy)).astype(np.uint8)
            #pdb.set_trace()
#	    skp = detector.detect(patch)
#            skp, sd = descriptor.compute(patch, skp)
#            desc.append(sd)
#            self.indx.append(len(skp))
#	return desc

    # --- For sift ---#
	def list2array(self,desc):
		nums = sum(self.indx)
		descs = np.empty((128,nums),dtype=np.float64)
		counts = 0
		for item in desc:
			if item == None:
				continue
				for j in range(item.shape[0]):
					descs[:,counts] = item[j,:].T
					counts += 1
		return descs.astype(np.float32)


	def best_match(self, img):
		results, dists = self.flann.nn_index(img)
		return self.warps[results[0]]