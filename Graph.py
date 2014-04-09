"""Author: Ankush Roy

	Input : trackerlines  -> File of tracked points
					groundtruthlines -> File of GT points
					TH -> Pixel threshold for succ
					t -> To simulate fast motion
"""

import math 
def main():
	frameRate = 1
	trackerlines = open("bookIIIRansac.txt",'r').readlines()
	groundtruthlines = open("/home/ankush/Folder/TrackingBenchmark/GitSources/Paper/DataCompare/GT/nl_bookIII_s3.txt",'r').readlines()
	AnalysisData = []
	TH = 1
	Success(trackerlines,groundtruthlines,frameRate)


def Success(T,GT,t):
	Error = []
	J = 1
	while J < len(GT):
		Tracker = T[J].strip().split()
		Groundtruth = GT[J].strip().split()
		Err = 0
		XTracker, YTracker = 0, 0
		X_GT, Y_GT = 0,0 
		for JJ in range(1,9,2):
			XTracker += float(Tracker[JJ])/4
			X_GT += float(Groundtruth[JJ])/4
		for JJ in range(2,9,2):
			YTracker += float(Tracker[JJ])/4
			Y_GT += float(Groundtruth[JJ])/4
		Err  = (XTracker  - X_GT)**2 + (YTracker - Y_GT)**2
		print math.sqrt(Err)
		J = J + 1

if __name__ == '__main__':
	main()
