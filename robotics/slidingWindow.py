def slidingWindow(img, m, n, th, step = 1):
	
	'''
	img = image
	m   = height of the sliding window
	n   = width of the sliding window
	th  = threshold for object to be present 
	step = step of the window progression (Default set to 1)
	
	'''
	M,N = img.shape
	POSSIBLE_POSITIONS = []
	START_M = 0 
	START_N = 0
	
	if m > M or n > N:
		print 'Select lower size of sliding windows'
		return 
	else:
		while START_M + m <= M:
			while START_N + n <= N:
				if check(img[START_M:START_M + m,START_N:START_N + n]) >= th:
					POSSIBLE_POSITIONS.append([START_M, START_M + m, START_N, START_N + n])
				START_N += step
			START_M += step
		return POSSIBLE_POSITIONS

