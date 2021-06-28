# import the necessary packages
import numpy as np
import imutils
import cv2

def bf_SIFT(img1, img2):
	orb = cv2.SIFT_create()
	kp1, des1 = orb.detectAndCompute(img1, None)
	kp2, des2 = orb.detectAndCompute(img2, None)

	# matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1, des2,k=2)

	# select good matches
	goods = []
	for  i,pair in enumerate(matches):
		try:
			m, n = pair
			if m.distance < 0.7*n.distance:
				goods.append(m)
		except ValueError:
			pass
	good = []
	for  m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append([m])
	
	# draw matches
	img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:100],None,flags=2)
	matchedVis = imutils.resize(img3, width=1000)
	cv2.imshow("Matched Keypoints", matchedVis)
	cv2.waitKey(0)

	# align image
	src_points = np.float32([kp1[m.queryIdx].pt for m in goods]).reshape(-1, 1, 2)
	dst_points = np.float32([kp2[m.trainIdx].pt for m in goods]).reshape(-1, 1, 2)
	m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
	aligned = cv2.warpPerspective(img1, m, (img2.shape[1], img2.shape[0]))
	
	return aligned