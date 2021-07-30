# import the necessary packages
from align_images import *
import numpy as np
import argparse
import imutils
import cv2
import timeit
start = timeit.default_timer()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image that we'll align to template")
ap.add_argument("-t", "--template", required=True,
	help="path to input template image")
args = vars(ap.parse_args())

# load the input image and template from disk
print("[INFO] loading images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])

# align the images
print("[INFO] aligning image...")
# match1 = bf_SIFT(image, template)
# match1 = ORB_BF(image, template)
# match1 = SIFT_FLAAN(image, template)
# match1 = SIFT_BF(image, template)
# match1 = AKAZE_BF(image, template)
# match1 = BRISK_BF(image, template)
# match1 = bf_matcher(image, template)
match1 = ORB_FLANN(image, template)

# resize both the aligned and template images so we can easily
# visualize them on our screen
aligned = imutils.resize(match1, width=700)
template = imutils.resize(template, width=700)

# our first output visualization of the image alignment will be a
# side-by-side comparison of the output aligned image and the
# template
stacked = np.hstack([aligned, template])

# our second image alignment visualization will be *overlaying* the
# aligned image on the template, that way we can obtain an idea of
# how good our image alignment is
overlay = template.copy()
output = aligned.copy()
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

print("[INFO] image aligned")
# show the two output image alignment visualizations
cv2.imshow("Image Alignment Stacked", stacked)
cv2.imshow("Image Alignment Overlay", output)
stop = timeit.default_timer()

print('Time: ', stop - start) 
cv2.waitKey(0)