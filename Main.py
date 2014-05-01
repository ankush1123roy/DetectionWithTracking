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

class StandaloneTrackingApp(InteractiveTrackingApp):
	""" A demo program that uses OpenCV to grab frames. """
    
	def __init__(self, vc, tracker, filename, tracker_name, path,initparam,nframe,name = 'vis'):
		InteractiveTrackingApp.__init__(self, tracker, filename, tracker_name,name)
		self.vc = vc
		self.path = path
		self.initparam = initparam
		self.nframe = nframe

	def run(self):

		i = 280
		while i <= self.nframe:
			img = cv2.imread(self.path+'%04d.jpg'%i)
			if img == None: 
				print('error loading image')
				break
			if not self.on_frame(img,i,self.initparam): break
			i += 1
		self.cleanup()

if __name__ == '__main__':
	coarse_tracker = NNTracker(10000, 2, res=(40,40), use_scv=True)
	#fine_tracker = ESMTracker(5, res=(40,40), use_scv=True)
	fine_tracker = BakerMatthewsICTracker(40, res = (40,40), use_scv = True)
	tracker = CascadeTracker([coarse_tracker, fine_tracker])
	filename = '/home/ankush/OriginalNN/NNTracker/src/NNTracker/'
	path = '/home/ankush/OriginalNN/NNTracker/src/Data/Liquor/img/'
	#initparam = [[372, 435],[421 ,433],[425, 488], [373, 490]]
	initparam = [[256.0, 152.0], [329.0, 152.0], [329.0, 362.0], [256.0, 362.0]] # Book III
	#initparam = [[581,104],[160,92],[138,414],[602,423]]                    # Metaio 4_1
	tracker_name = 'DeerCVPR'
	nframe = 1500
	app = StandaloneTrackingApp(None, coarse_tracker,filename,tracker_name,path,initparam,nframe)
	app.run()
	app.cleanup()
