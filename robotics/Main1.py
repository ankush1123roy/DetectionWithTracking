"""
A small standalone application for tracker demonstration. Depends
on OpenCV VideoCapture to grab frames from the camera.

Author: Travis Dick (travis.barry.dick@gmail.com)
"""
import time
from BakerMatthewsICTracker import *
from CascadeTracker import *
from ESMTracker import *
from Homography import *
from InteractiveTracking1 import *
from MultiProposalTracker import *
from NNTracker1 import *
from ParallelTracker import *
from SiftDetection import *
import cv2
import cv
class StandaloneTrackingApp(InteractiveTrackingApp):
	""" A demo program that uses OpenCV to grab frames. """

	def __init__(self, path, vc, tracker, filename, tracker_name, nframe,name = 'vis'):
		InteractiveTrackingApp.__init__(self, tracker, filename, tracker_name,name)
		self.vc = vc
		self.nframe = nframe
		self.path = path

	def run(self):

		i = 1
		capture = cv2.VideoCapture('bookIIIM.avi')
		#camera =  cv2.VideoCapture(0)
		
		while i <= self.nframe:
			flag, frame = capture.read()
			#f, frame = camera.read()
			#frame = cv2.imread(self.path+'%010d.ppm'%i)
			if frame == None: 
				print('error loading image')
				break
			if not self.on_frame(frame,i,self.initparam): break
			i += 1
		self.cleanup()

if __name__ == '__main__':
	coarse_tracker = NNTracker(1000, 2, res=(40,40), use_scv=True)
	fine_tracker = ESMTracker(5, res=(40,40), use_scv=True)
	#fine_tracker = BakerMatthewsICTracker(40, res = (40,40), use_scv = True)
	tracker = CascadeTracker([coarse_tracker, fine_tracker])
	filename = '/home/ankush/OriginalNN/NNTracker/src/NNTracker/'
	tracker_name = 'DeerCVPR'
	nframe = 4000
	app = StandaloneTrackingApp(path, None, coarse_tracker,filename,tracker_name,nframe)
	app.run()
	app.cleanup()
