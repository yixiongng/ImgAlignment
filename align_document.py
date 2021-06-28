# import the necessary packages
from pyimagesearch.alignment.align_images import *
import numpy as np
import argparse
import imutils
import cv2

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
print("[INFO] aligning images...")
# aligned = align_images(image, template, debug=True)

# # resize both the aligned and template images so we can easily
# # visualize them on our screen
# aligned = imutils.resize(aligned, width=700)
# template = imutils.resize(template, width=700)

# # our first output visualization of the image alignment will be a
# # side-by-side comparison of the output aligned image and the
# # template
# stacked = np.hstack([aligned, template])

# # our second image alignment visualization will be *overlaying* the
# # aligned image on the template, that way we can obtain an idea of
# # how good our image alignment is
# overlay = template.copy()
# output = aligned.copy()
# cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

# # show the two output image alignment visualizations
# cv2.imshow("Image Alignment Stacked", stacked)
# cv2.imshow("Image Alignment Overlay", output)
# cv2.waitKey(0)

# img = get_corrected_img(image, template)
# cv2.imshow('Corrected image', img)
# cv2.waitKey()

# match = bf_matcher(image, template)
# cv2.imshow('Matches', match)
# cv2.waitKey()
match1 = bf_SIFT(image, template)
aligned = imutils.resize(match1, width=700)
template = imutils.resize(template, width=700)
stacked = np.hstack([aligned, template])
overlay = template.copy()
output = aligned.copy()
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

cv2.imshow("Image Alignment Stacked", stacked)
cv2.imshow("Image Alignment Overlay", output)
cv2.waitKey(0)