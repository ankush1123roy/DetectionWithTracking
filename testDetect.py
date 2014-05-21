import numpy as np
from featuresUnSup import *
import cv2


img1 = cv2.cvtColor(cv2.imread('../Template1/' + '0001' + '.jpg'), cv2.COLOR_BGR2GRAY) # trainImage
img2 = cv2.cvtColor(cv2.imread('/home/ankush/OriginalNN/NNTracker/src/Data/Liquor/img/0001.jpg'), cv2.COLOR_BGR2GRAY) # testImage

F = features('codes.txt', 'mean.txt', 'whiten.txt')
XC,XF = F.extract_features(img1)
import pdb;pdb.set_trace()
