# import the necessary packages
import numpy as np
import imutils
import cv2
def align_images(image, template, maxFeatures=500, keepPercent=0.75, debug=False):
	# convert both the input image and template to grayscale
	imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # use ORB to detect keypoints and extract (binary) local
	# invariant features
	orb = cv2.ORB_create(maxFeatures)
	(kpsA, descsA) = orb.detectAndCompute(imageGray, None)
	(kpsB, descsB) = orb.detectAndCompute(templateGray, None)
	# match the features
	method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
	matcher = cv2.DescriptorMatcher_create(method)
	matches = matcher.match(descsA, descsB, None)
    # sort the matches by their distance (the smaller the distance,
	# the "more similar" the features are)
	matches = sorted(matches, key=lambda x:x.distance)
	# keep only the top matches
	keep = int(len(matches) * keepPercent)
	matches = matches[:keep]
	# check to see if we should visualize the matched keypoints
	if debug:
		matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
			matches, None)
		matchedVis = imutils.resize(matchedVis, width=1000)
		cv2.imshow("Matched Keypoints", matchedVis)
		cv2.waitKey(0)
    # allocate memory for the keypoints (x, y)-coordinates from the
	# top matches -- we'll use these coordinates to compute our
	# homography matrix
	ptsA = np.zeros((len(matches), 2), dtype="float")
	ptsB = np.zeros((len(matches), 2), dtype="float")
	# loop over the top matches
	for (i, m) in enumerate(matches):
		# indicate that the two keypoints in the respective images
		# map to each other
		ptsA[i] = kpsA[m.queryIdx].pt
		ptsB[i] = kpsB[m.trainIdx].pt
    # compute the homography matrix between the two sets of matched
	# points
	(H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
	# use the homography matrix to align the images
	(h, w) = template.shape[:2]
	aligned = cv2.warpPerspective(image, H, (w, h))
	# return the aligned image
	return aligned

def get_corrected_img(img1, img2):
    MIN_MATCHES = 40

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

    if len(good_matches) > MIN_MATCHES:
        src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        corrected_img = cv2.warpPerspective(img1, m, (img2.shape[1], img2.shape[0]))

        return corrected_img
    return img1

def bf_matcher(img1, img2):
	orb = cv2.ORB_create(nfeatures=500)
	kp1, des1 = orb.detectAndCompute(img1, None)
	kp2, des2 = orb.detectAndCompute(img2, None)

	# matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1, des2)
	matches = sorted(matches, key=lambda x: x.distance)
	# draw first 50 matches
	match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
	matchedVis = imutils.resize(match_img, width=1000)
	# cv2.imshow("Matched Keypoints", matchedVis)
	# cv2.waitKey(0)
	# ptsA = np.zeros((len(matches), 2), dtype="float")
	# ptsB = np.zeros((len(matches), 2), dtype="float")
	# # loop over the top matches
	# for (i, m) in enumerate(matches):
	# 	# indicate that the two keypoints in the respective images
	# 	# map to each other
	# 	ptsA[i] = kp1[m.queryIdx].pt
	# 	ptsB[i] = kp2[m.trainIdx].pt
    # # compute the homography matrix between the two sets of matched
	# # points
	# (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
	# # use the homography matrix to align the images
	# (h, w) = img1.shape[:2]
	# aligned = cv2.warpPerspective(img2, H, (w, h))
	# return the aligned image
	return matchedVis

def bf_SIFT(img1, img2):
	orb = cv2.SIFT_create()
	kp1, des1 = orb.detectAndCompute(img1, None)
	kp2, des2 = orb.detectAndCompute(img2, None)

	# matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1, des2,k=2)
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
	img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:100],None,flags=2)
	matchedVis = imutils.resize(img3, width=1000)
	cv2.imshow("Matched Keypoints", matchedVis)
	cv2.waitKey(0)
	src_points = np.float32([kp1[m.queryIdx].pt for m in goods]).reshape(-1, 1, 2)
	dst_points = np.float32([kp2[m.trainIdx].pt for m in goods]).reshape(-1, 1, 2)
	m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
	aligned = cv2.warpPerspective(img1, m, (img2.shape[1], img2.shape[0]))
	
	return aligned