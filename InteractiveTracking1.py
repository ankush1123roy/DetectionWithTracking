"""
Author: Travis Dick (travis.barry.dick@gmail.com)
"""
import os
import cv
import cv2
import numpy as np

from CascadeTracker import *
from Homography import *
from ImageUtils import *
from NNTracker1 import *
from SiftDetection import *

class InteractiveTrackingApp:
	def __init__(self, tracker, filename=None, tracker_name=None,name="vis"):
		""" 
			An interactive window for initializing and visualizing tracker state.
			The on_frame method should be called for each new frame. Typically real
			applications subclass InteractiveTrackingApp and build in some application
			loop that captures frames and calls on_frame.

			Parameters:
			-----------
			tracker : TrackerBase
			Any class implementing the interface of TrackerBase. 

			ame : string
			The name of the window. Due to some silliness in OpenCV this must
			be unique (in the set of all OpenCV window names).

			See Also:
			---------
			StandaloneTrackingApp
			RosInteractiveTrackingApp
		"""

		self.tracker = tracker
		self.name = name
		self.m_start = None
		# New definition
		self.m_ur = None
		self.m_ll = None
		self.m_end = None
	# New definition
		self.initparamtemp = []
		self.initparam = None
	# End of new definition 
		self.gray_img = None
		self.paused = False
		self.img = None
		self.times = 1
		self.filename = filename
		self.height = None
		self.width = None
		self.visible = True
		#self.templateBuffer = [0]*20
		
		#if not os.path.exists(filename):
		#	os.mkdir(filename)
		self.fname = open(filename+'/'+tracker_name+'.txt','w')
		self.fname.write('%-8s%-8s%-8s%-8s%-8s%-8s%-8s%-8s%-8s\n'%('frame','ulx','uly','urx','ury','lrx','lry','llx','lly'))
		cv2.namedWindow(self.name)
		self.D = Detector( 0.9, 10)
		self.D.train()
			#cv2.setMouseCallback(self.name, self.mouse_handler4)
	#self.writer = cv2.VideoWriter('alpha.avi',cv.CV_FOURCC('D','I','V','3'),10,size)

	def display(self, img):
		annotated_img = img.copy()
		original_img = img.copy()
		if self.tracker.is_initialized():
			corners = self.tracker.get_region()
			img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			#print img.shape
			#if Flag == 1:
			#	draw_region(annotated_img, corners, (0,255,255), 2)
			#else:
			draw_region(annotated_img, corners, (0,255,0), 2)
			#import pdb;pdb.set_trace()
			self.fname.write('%-15s%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f\n'%('frame'+('%05d'%(self.times))+'.jpg',corners[0,0],corners[1,0],corners[0,1],corners[1,1],corners[0,2],corners[1,2],corners[0,3],corners[1,3]))
#	'''
#	if self.m_start != None and self.m_end != None:
#            ul = (min(self.m_start[0],self.m_end[0]), min(self.m_start[1],self.m_end[1]))
#            lr = (max(self.m_start[0],self.m_end[0]), max(self.m_start[1],self.m_end[1]))           #             corners = np.array([ ul, [lr[0],ul[1]], lr, [ul[0],lr[1]]]).T
#	'''
		elif len(self.initparamtemp) == 4:
			corners = self.initparamtemp
			draw_region(annotated_img, corners, (255,0,0), 1)
			self.fname.write('%-15s%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f\n'%('frame'+('%05d'%(self.times))+'.jpg',corners[0,0],corners[1,0],corners[0,1],corners[1,1],corners[0,2],corners[1,2],corners[0,3],corners[1,3]))
		#cv2.imshow(self.name, annotated_img)
		#import pdb;pdb.set_trace()
		cv2.imwrite('../{0:04d}.jpg'.format(self.times),annotated_img)
		temp = original_img[corners[1][1]:corners[1][2], corners[0][0]:corners[0][1]]
		#import pdb;pdb.set_trace()
		#self.templateBuffer.append(temp)
		#self.templateBuffer.pop(0)
		
		
		cv.WaitKey(500)
	#if self.times == 1: cv.WaitKey(6000)
	#self.writer.write(annotated_img)
		#cv.SaveImage( self.filename+'/'+'%04d'%self.times+'.jpg',cv.fromarray(annotated_img))

	
	def mouse_handler(self, evt,x,y,arg,extra):
		if self.gray_img == None: return 
		if evt == cv2.EVENT_LBUTTONDOWN and self.m_start == None:
			self.m_start = (x,y)
			self.m_end = (x,y)
			self.paused = True
		elif evt == cv2.EVENT_MOUSEMOVE and self.m_start != None:
			self.m_end = (x,y)
		elif evt == cv2.EVENT_LBUTTONUP:
			self.m_end = (x,y)
			ul = (min(self.m_start[0],self.m_end[0]), min(self.m_start[1],self.m_end[1]))
			lr = (max(self.m_start[0],self.m_end[0]), max(self.m_start[1],self.m_end[1]))
			self.tracker.initialize_with_rectangle(self.gray_img, ul, lr)
			self.m_start, self.m_end = None, None
			self.paused = False
			self.inited = True
	#cv.WaitKey(1000)

	def mouse_handler4(self,evt,x,y,arg,extra):
		if self.gray_img == None: return
		if evt == cv2.EVENT_LBUTTONDOWN and len(self.initparamtemp) <= 3:
			self.initparamtemp.append([x,y])
			self.paused = True
		elif evt == cv2.EVENT_LBUTTONDOWN and len(self.initparamtemp) == 4:
			self.initparamtemp = []
			self.initparamtemp.append([x,y])
			self.paused = True
		if len(self.initparamtemp) == 4:
			self.initparam = np.array(self.initparamtemp).T
			self.tracker.initialize(self.gray_img,self.initparam)
			self.paused = False
			self.inited = True
	
	def on_frame(self, img, numtimes,initparamtemp):
		print(numtimes)
		self.times = numtimes
		if numtimes == 1:
			#cv.WaitKey(6000)	
			#self.initparamtemp = [[336,165],[362,165],[362,226],[336,226]]
			self.initparamtemp = initparamtemp
			self.initparam = np.array(self.initparamtemp).T
			self.gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			self.gray_img = self.gray_img.astype(np.float64)
			self.tracker.initialize(self.gray_img,self.initparam)
			self.paused = False
			self.inited = True
			'''
			if len(self.initparamtemp) == 4:	
			corners = 
			self.fname.write('%-15s%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f\n'%('frame'+('%05d'%(self.times))+'.jpg',corners[0,0],corners[1,0],corners[0,1],corners[1,1],corners[0,2],corners[1,2],corners[0,3],corners[1,3]))
			'''
		if self.visible == True:
			corners = self.tracker.get_region()
		else:
			corners = self.D.Detect(img)
			if corners == False:
				pass
			else:
				self.visible = True
		#self.width = (corners[0][1] - corners[0][0]) / 2
		#self.height = (corners[1][2] - corners[1][1]) / 2 
		#print self.height, self.width
		if not self.paused:
			self.img = img
			self.gray_img = cv2.GaussianBlur(to_grayscale(img), (5,5), 3)
			self.gray_img = to_grayscale(img)
			val1, val2 = self.tracker.update(self.gray_img)
			if val2 == False:
			
			#import pdb;pdb.set_trace()
			#X, Y =  self.D.Detect(img)
			#self.initparamtemp = [[int(X) - self.width, int(Y) - self.height],[int(X) + self.width, int(Y) - self.height],[int(X) + self.width, self.height + int(Y)],[int(X) - self.width, int(Y) + self.height]]
			
			#cv2.imwrite('/home/ankush/OriginalNN/NNTracker/src/NNTracker/src/{0:04d}.jpg'.format(self.times), self.templateBuffer[-1])
			
				self.initparamtemp = self.D.Detect(img)
			#import pdb;pdb.set_trace()
			#print self.initparamtemp
			#if self.initparamtemp != False:
			# Change this initialisation point
				if self.initparamtemp == False:
					self.visible = False
				else:
					self.initparam = np.array(self.initparamtemp).T
					self.gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
					self.gray_img = self.gray_img.astype(np.float64)
					self.tracker.initialize(self.gray_img,self.initparam)
			#else:
			#	print 'Low Keypoints'
				
		if self.img != None: self.display(self.img)
		key = cv.WaitKey(7)
		if key == ord(' '): self.paused = not self.paused
		elif key > 0: return False
		self.times = self.times + 1
		return True

	def cleanup(self):
		#cv2.destroyWindow(self.name)
	#pass
		self.fname.close()
	#self.writer.release
