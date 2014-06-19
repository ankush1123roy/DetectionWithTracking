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
class StandaloneTrackingApp(InteractiveTrackingApp):
	""" A demo program that uses OpenCV to grab frames. """

	def __init__(self, vc, tracker, filename, tracker_name, path,initparam,nframe,name = 'vis'):
		InteractiveTrackingApp.__init__(self, tracker, filename, tracker_name,name)
		self.vc = vc
		self.path = path
		self.initparam = initparam
		self.nframe = nframe

	def run(self):

		i = 1
		capture = cv2.VideoCapture('VID_20140619_022236.mp4')
		#camera =  cv2.VideoCapture(0);
		while i <= self.nframe:
			flag, frame = capture.read()
			#f, img = camera.read()
			if frame == None: 
				print('error loading image')
				break
			if not self.on_frame(frame,i,self.initparam): break
			i += 1
		self.cleanup()

if __name__ == '__main__':
	coarse_tracker = NNTracker(10000, 2, res=(40,40), use_scv=True)
	#fine_tracker = ESMTracker(5, res=(40,40), use_scv=True)
	fine_tracker = BakerMatthewsICTracker(40, res = (40,40), use_scv = True)
	tracker = CascadeTracker([coarse_tracker, fine_tracker])
	filename = '/home/ankush/OriginalNN/NNTracker/src/NNTracker/'
	path = '/home/ankush/OriginalNN/NNTracker/src/Data/nl_cereal_s5/frame'
	#initparam = [[372, 435],[421 ,433],[425, 488], [373, 490]]
	initparam = [[368.0, 439.0], [420.0, 440.0], [419.0, 491.0], [367.0, 490.0]] # Book III
	#initparam = [[581,104],[160,92],[138,414],[602,423]]                    # Metaio 4_1
	tracker_name = 'DeerCVPR'
	nframe = 4000
	app = StandaloneTrackingApp(None, coarse_tracker,filename,tracker_name,path,initparam,nframe)
	app.run()
	app.cleanup()
