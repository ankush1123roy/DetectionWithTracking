import numpy as np
from pylab import imshow, figure, show, hold, ion, ioff, draw

def visualize(centroids, H, W=None):

    if W == None:
        W = H
    N = centroids.shape[1] / (H * W)
    
    K = centroids.shape[0]
    COLS = round(np.sqrt(K))
    ROWS = np.ceil(K / COLS)
    COUNT=COLS * ROWS

    image = np.ones((ROWS*(H+1), COLS*(W+1), N)) * 0.15
    for i in range(K):

        r = np.floor(i / COLS)
        c = np.mod(i, COLS)
        image[(r*(H+1)):((r+1)*(H+1))-1,(c*(W+1)):((c+1)*(W+1))-1,:] = np.reshape(centroids[i,:], (H,W,N))

    mn = -0.35 #0.35 default
    mx = 0.35
    image = (image - mn) / (mx - mn)

    ion()
    #figure(1)
    imshow(image)
    #show()
    draw()
    #ioff()
