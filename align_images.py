# import the necessary packages
import numpy as np
import imutils
import cv2
import glob
# all passed, avg = 5s
def bf_SIFT(image, template):
	orb = cv2.SIFT_create()
	kp1, des1 = orb.detectAndCompute(image, None)
	kp2, des2 = orb.detectAndCompute(template, None)

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
	img3 = cv2.drawMatchesKnn(image, kp1, template, kp2, good[:100],None,flags=2)
	matchedVis = imutils.resize(img3, width=1000)
	cv2.imshow("Matched Keypoints", matchedVis)
	# cv2.waitKey(0)

	# align image
	src_points = np.float32([kp1[m.queryIdx].pt for m in goods]).reshape(-1, 1, 2)
	dst_points = np.float32([kp2[m.trainIdx].pt for m in goods]).reshape(-1, 1, 2)
	m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
	aligned = cv2.warpPerspective(image, m, (template.shape[1], template.shape[0]))
	
	return aligned
# 2 skewed, 1 crash, 5 passed, avg = 1.5s
def ORB_BF(image, template):
	orb = cv2.ORB_create()
	kp1, des1 = orb.detectAndCompute(image,None)
	kp2, des2 = orb.detectAndCompute(template,None)
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)
	good = []
	for  m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append([m])
	goods = []
	for  i,pair in enumerate(matches):
		try:
			m, n = pair
			if m.distance < 0.7*n.distance:
				goods.append(m)
		except ValueError:
			pass
	
	# draw matches
	img3 = cv2.drawMatchesKnn(image, kp1, template, kp2, good[:100],None,flags=2)
	matchedVis = imutils.resize(img3, width=1000)
	cv2.imshow("Matched Keypoints", matchedVis)
	# cv2.waitKey(0)
	src_points = np.float32([kp1[m.queryIdx].pt for m in goods]).reshape(-1, 1, 2)
	dst_points = np.float32([kp2[m.trainIdx].pt for m in goods]).reshape(-1, 1, 2)
	m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
	aligned = cv2.warpPerspective(image, m, (template.shape[1], template.shape[0]))
	
	return aligned
# all passed, avg = 4.5s
def SIFT_FLAAN(image, template):
	sift = cv2.SIFT_create()
	kp1, des1 = sift.detectAndCompute(image,None)
	kp2, des2 = sift.detectAndCompute(template,None)
	FLANNINDEXKDTREE = 1
	indexparams = dict(algorithm = FLANNINDEXKDTREE, trees = 5)
	searchparams = dict(checks=50)
	flann = cv2.FlannBasedMatcher(indexparams,searchparams)
	matches = flann.knnMatch(des1,des2,k=2)
	good = []
	for  m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append([m])
	goods = []
	for  i,pair in enumerate(matches):
		try:
			m, n = pair
			if m.distance < 0.7*n.distance:
				goods.append(m)
		except ValueError:
			pass
	
	# draw matches
	img3 = cv2.drawMatchesKnn(image, kp1, template, kp2, good[:100],None,flags=2)
	matchedVis = imutils.resize(img3, width=1000)
	cv2.imshow("Matched Keypoints", matchedVis)
	# cv2.waitKey(0)
	src_points = np.float32([kp1[m.queryIdx].pt for m in goods]).reshape(-1, 1, 2)
	dst_points = np.float32([kp2[m.trainIdx].pt for m in goods]).reshape(-1, 1, 2)
	m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
	aligned = cv2.warpPerspective(image, m, (template.shape[1], template.shape[0]))
	
	return aligned
# all passed, avg = 13s
def SIFT_BF(image, template):
	img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

	#sift
	sift = cv2.SIFT_create()

	keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
	keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

	#feature matching
	bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

	matches = bf.match(descriptors_1,descriptors_2)
	matches = sorted(matches, key = lambda x:x.distance)

	img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
	matchedVis = imutils.resize(img3, width=1000)
	cv2.imshow("Matched Keypoints", matchedVis)
	# cv2.waitKey(0)
	src_points = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
	dst_points = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
	m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
	aligned = cv2.warpPerspective(image, m, (template.shape[1], template.shape[0]))
	
	return aligned
# all passed, avg = 3.2s
def AKAZE_BF(image, template):
	akaze = cv2.AKAZE_create()
	kp1, des1 = akaze.detectAndCompute(image, 0)
	kp2, des2 = akaze.detectAndCompute(template, 0)
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)

	goods = []
	for  i,pair in enumerate(matches):
		try:
			m, n = pair
			if m.distance < 0.7*n.distance:
				goods.append(m)
		except ValueError:
			pass
	img3 = cv2.drawMatches(image, kp1, template, kp2, goods[:50], template, flags=2)
	matchedVis = imutils.resize(img3, width=1000)
	cv2.imshow("Matched Keypoints", matchedVis)
	# cv2.waitKey(0)
	src_points = np.float32([kp1[m.queryIdx].pt for m in goods]).reshape(-1, 1, 2)
	dst_points = np.float32([kp2[m.trainIdx].pt for m in goods]).reshape(-1, 1, 2)
	m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
	aligned = cv2.warpPerspective(image, m, (template.shape[1], template.shape[0]))
	
	return aligned
# all passed, avg = 3.5s
def BRISK_BF(image, template):
	brisk = cv2.BRISK_create()
	kp1, des1 = brisk.detectAndCompute(image,None)
	kp2, des2 = brisk.detectAndCompute(template,None)
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)

	goods = []
	for  i,pair in enumerate(matches):
		try:
			m, n = pair
			if m.distance < 0.7*n.distance:
				goods.append(m)
		except ValueError:
			pass
	img3 = cv2.drawMatches(image, kp1, template, kp2, goods[:50], template, flags=2)
	matchedVis = imutils.resize(img3, width=1000)
	cv2.imshow("Matched Keypoints", matchedVis)
	# cv2.waitKey(0)
	src_points = np.float32([kp1[m.queryIdx].pt for m in goods]).reshape(-1, 1, 2)
	dst_points = np.float32([kp2[m.trainIdx].pt for m in goods]).reshape(-1, 1, 2)
	m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
	aligned = cv2.warpPerspective(image, m, (template.shape[1], template.shape[0]))
	
	return aligned
# all passed, avg = 1.55s
def bf_matcher(image, template):
	orb = cv2.ORB_create(nfeatures=500)
	kp1, des1 = orb.detectAndCompute(image, None)
	kp2, des2 = orb.detectAndCompute(template, None)

	# matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1, des2)
	matches = sorted(matches, key=lambda x: x.distance)
	img3 = cv2.drawMatches(image, kp1, template, kp2, matches[:50], template, flags=2)
	matchedVis = imutils.resize(img3, width=1000)
	cv2.imshow("Matched Keypoints", matchedVis)
	# cv2.waitKey(0)
	src_points = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
	dst_points = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
	m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
	aligned = cv2.warpPerspective(image, m, (template.shape[1], template.shape[0]))
	
	return aligned
# failed 1, 7 passed, avg = 1.5s
def ORB_FLANN(img1, img2):
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # As per Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
        	good_matches.append(m)
    src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    corrected_img = cv2.warpPerspective(img1, m, (img2.shape[1], img2.shape[0]))

    return corrected_img