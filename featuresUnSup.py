import numpy as np
from visual import *

class features:
	
	def __init__(self,
							codes,
							mean,
							whiten,
							rfsize = [5,5,1],
							npatches = 10000,
							beta = 10,
							gamma = 0.1,
							pyramid = 3,
							alpha = 0,
							stride = 4,
							nfeats = 40,
							dimsize = [16, 16],
							conv = False,
							tilesize = [16,16]):


		self.rfsize = rfsize
		self.D_codes = np.array(self.codes((open(codes,'r').readlines())))
		self.D_mean = np.array(self.mean(open(mean,'r').readlines()))
		self.D_whiten = np.array(self.whiten(open(whiten,'r').readlines()))
		self.beta = beta
		self.gamma = gamma
		self.pyramid = pyramid
		self.alpha = alpha
		self.stride = stride
		self.dimsize = dimsize
		self.tilesize = tilesize
		self.conv = conv
		self.nfeats = nfeats
		self.npatches = npatches

	def codes(self, code):
		CODES = []
		for i in range(len(code)):
			CODES.append([float(i) for i in code[i].strip().split()])
		return CODES
			
		
	def whiten(self, whiten):
		WHITEN = []
		for i in range(len(whiten)):
			WHITEN.append([float(i) for i in whiten[i].strip().split()])
		
		return WHITEN
		
	def mean(self, mean):
		MEAN = [float(i) for i in mean[0].strip().split()]
		return MEAN

	def extract_features(self, images):
		if self.pyramid == 2:
			mapprod = 5
		elif self.pyramid == 3:
			mapprod = 14
		XC = np.zeros((len(images), mapprod * self.nfeats))
		XL = [0]*len(images)
		XF = [0]*len(images)
		prows = self.tilesize[0] - self.rfsize[0] + 1
		pcols = self.tilesize[1] - self.rfsize[1] + 1
		
		# Loop of each image
		#for i in range(len(images)):
			
			# Grab an image
		im = images
		if self.conv:
			prows = im.shape[0] - self.rfsize[0] + 1
			pcols = im.shape[1] - self.rfsize[1] + 1
				
		(subregions, rowinds, colinds) = self.window(im)
		features = np.zeros((prows * pcols * self.nfeats, subregions.shape[0]))
			
			# Extract sub features
		for j in range(subregions.shape[0]):
			features[:,j] = self.extract_subfeatures(subregions[j,:].reshape(-1,1).T, im)
			
			# Reshape into spatial regions
		index = 0
		field = np.zeros((prows * len(rowinds), pcols * len(colinds), self.nfeats))
		for j in range(len(rowinds)):
			for k in range(len(colinds)):
				field[j * prows : (j+1) * prows, k * pcols : (k+1) * pcols, :] = np.reshape(features[:,index], (prows, pcols, self.nfeats))
				index = index + 1
					
			# Pooling
		middle = self.pooling(field.flatten(), prows * len(rowinds), pcols * len(colinds), self.nfeats, (2, 2)).reshape(-1, 1)
		top = self.pooling(field.flatten(), prows * len(rowinds), pcols * len(colinds), self.nfeats, (1, 1)).reshape(-1, 1)
		if self.pyramid == 3:
			bottom = self.pooling(field.flatten(), prows * len(rowinds), pcols * len(colinds), self.nfeats, (3, 3)).reshape(-1, 1)
			XC = np.c_[np.c_[bottom.T, middle.T], top.T]
		elif self.pyramid == 2:
			XC = np.c_[middle.T, top.T]
		L = self.pooling(field.flatten(), prows * len(rowinds), pcols * len(colinds), self.nfeats, (self.dimsize[0], self.dimsize[1]))
		XF = field
			
		return (XC, XF)




	def extract_subfeatures(self, X, im):
		'''
		Compute sub feature maps of flattened sub - regions
		'''
		if self.conv:
			tile = np.shape(im)
		else:
			tile = self.tilesize
		k = self.D_codes.shape[0]
		prows = tile[0]  - self.rfsize[0] + 1
		pcols = tile[1]  - self.rfsize[1] + 1
		r = self.rfsize[0] * self.rfsize[1] * self.rfsize[2]
		f = k * prows * pcols
		XC = np.zeros((X.shape[0], f))
		
		# Loop over each region
		for i in range(X.shape[0]):
			
			#Extract all patches
			patches = self.patches_all(X[i:], tile)
			
			#Contrast Normalisation
			pathces = self.local_contrast_normalization(patches)
			
			#Apply whitening
			#import pdb;pdb.set_trace() 
			patches = np.dot(patches - self.D_mean.T, self.D_whiten)
			
			# Feature Encoding
			xc = np.dot(patches, self.D_codes.T) - self.alpha
			patches = xc
			patches = np.array(patches)
			
			# Map contrast normalise
			patches = self.map_normalization(patches)
			
			#Grab the features
			patches = np.reshape(patches, (prows, pcols, k))
			XC[i,:] = patches.flatten()
		return XC
			
			
	def window(self, im):
		'''
		Extract 'tiled' region of the image
		'''
		
		if self.conv:
			tile = np.shape(im)
		else:
			tile = self.tilesize
		rowinds = np.arange(0, im.shape[0] - tile[0] + 1, self.stride)
		colinds = np.arange(0, im.shape[1] - tile[1] + 1, self.stride)
		if len(im.shape) == 3:
			subregions = np.zeros((len(rowinds) * len(colinds), tile[0] * tile[1] * im.shape[2]))
		else:
			subregions = np.zeros((len(rowinds) * len(colinds), tile[0] * tile[1]))
		index = 0
		
		for i in rowinds:
			for j in colinds:
				sr = im[i : i + tile[0], j : j + tile[1]]
				subregions[index, :] = sr.flatten('F')
				index = index + 1
		return (subregions, rowinds, colinds)
		
		
		
	def patches_all(self, X, tile):
		'''
		Get (valid) image patches into 'rows' of patches 
		X is a vector of a flatten image sub region
		'''
		count = 0
		ims = tile[0] * tile[1]
		p = np.reshape(X[:ims], (tile[0], tile[1]), 'F')
		patches = self.im2col(p, tile)
		if len(tile) == 3:
			for j in range(1, tile[2]):
				import pdb;pdb.set_trace()
				p = np.reshape(X[j * ims : (j + 1) * ims], (tile[0], tile[1]), 'F')
				patches = np.r_[patches, self.im2col(p, tile)]
		return patches.T
				
		
	def im2col(self , p, tile):
		'''
		Extract columns of image blocks of (rfsize, rfsize)
		'''
		
		prows = tile[0] - self.rfsize[0] + 1
		pcols = tile[1] - self.rfsize[1] + 1
		imcol = np.zeros((self.rfsize[0] * self.rfsize[1], prows * pcols))
		count = 0
		for x in range(prows):
			for y in range(pcols):
				imcol[:,count] = p[x : x + self.rfsize[0], y : y + self.rfsize[1]].flatten()
				count += 1
		return imcol

	def map_normalization(self, patches):
		'''
		Normalize encoded feature map patches
		'''
		mn = np.mean(patches, 1).reshape(-1,1)
		sd = np.std(patches, 1).reshape(-1, 1)
		patches = (patches - mn) / np.maximum(np.mean(sd).reshape(-1, 1), sd)
		patches[np.isnan(patches)] = 0
		return patches
		
		
	def local_contrast_normalization(self, patches):
		'''
		Individual patch based contrast normalization
		'''
		patch_mean = np.mean(patches, 1).reshape(-1, 1)
		patch_std = np.sqrt(np.var(patches, 1) + self.beta).reshape(-1, 1)
		return (patches - patch_mean) / patch_std
		
		
			
	def pooling(self, XC, nrows, ncols, nmaps, gridsize):
		'''
		Apply (gridsize x gridsize) pooling over a flattened image vector XC
		'''
		Q = np.zeros((gridsize[0], gridsize[1], nmaps))
		im = np.reshape(XC, (nrows, ncols, nmaps))
		r = gridsize[0] * np.ceil(1.0*nrows / gridsize[0])
		c = gridsize[1] * np.ceil(1.0*ncols / gridsize[1])
		padval_r = np.ceil(1.0*(r - nrows) / 2)
		padval_c = np.ceil(1.0*(c - ncols) / 2)
		tmp = np.zeros((im.shape[0] + 2 * padval_r, im.shape[1] + 2 * padval_c, im.shape[2]))
		tmp[padval_r : im.shape[0] + padval_r, padval_c : im.shape[1] + padval_c, :] = im
		im = tmp[:r, :c, :]
		x = np.split(np.arange(r), gridsize[0])
		y = np.split(np.arange(c), gridsize[1])
		for j in range(gridsize[0]):
			for k in range(gridsize[1]):
				region = im[x[j][0] : x[j][-1] + 1, y[k][0] : y[k][-1] + 1, :]
				Q[j,k,:] = np.sum(np.sum(region, 0), 0)
		return Q.flatten()
				
		
			
