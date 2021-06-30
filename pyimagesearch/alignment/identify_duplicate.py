from __future__ import division

import cv2
import numpy as np
import glob
import pandas as pd


listOfTitles1 = []
listOfTitles2 = []
listOfSimilarities = []

    # Sift and Flann
sift = cv2.xfeatures2d.SIFT_create()


index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Load all the images1

countInner = 0
countOuter = 1

folder = r"/Downloads/images/**/*"
folder = "SiftImages/*"


siftOut = {}
for a in glob.iglob(folder,recursive=True):
    if not a.lower().endswith(('.jpg','.png','.tif','.tiff','.gif')):
        continue
    image1 = cv2.imread(a)
    kp_1, desc_1 = sift.detectAndCompute(image1, None)
    siftOut[a]=(kp_1,desc_1)



for a in glob.iglob(folder,recursive=True):
    if not a.lower().endswith(('.jpg','.png','.tif','.tiff','.gif')):
        continue

    (kp_1,desc_1) = siftOut[a]

    for b in glob.iglob(folder,recursive=True):


        if not b.lower().endswith(('.jpg','.png','.tif','.tiff','.gif')):

            continue

        if b.lower().endswith(('.jpg','.png','.tif','.tiff','.gif')):

            countInner += 1


        print(countInner, "", countOuter)

        if countInner <= countOuter:

            continue

        #### image1 = cv2.imread(a)
        #### kp_1, desc_1 = sift.detectAndCompute(image1, None)
        ####
        #### image2 = cv2.imread(b)
        #### kp_2, desc_2 = sift.detectAndCompute(image2, None)

        (kp_2,desc_2) = siftOut[b]

        matches = flann.knnMatch(desc_1, desc_2, k=2)

        good_points = []

        if good_points == 0:

            continue

        for m, n in matches:
            if m.distance < 0.6*n.distance:
                good_points.append(m)

        number_keypoints = 0
        if len(kp_1) >= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        percentage_similarity = float(len(good_points)) / number_keypoints * 100

        listOfSimilarities.append(str(int(percentage_similarity)))
        listOfTitles2.append(b)

        listOfTitles1.append(a)

    countInner = 0
    if a.lower().endswith(('.jpg','.png','.tif','.tiff','.gif')):
        countOuter += 1

zippedList =  list(zip(listOfTitles1,listOfTitles2, listOfSimilarities))

print(zippedList)

dfObj = pd.DataFrame(zippedList, columns = ['Original', 'Title' , 'Similarity'])

### dfObj.to_csv(r"/Downloads/images/DuplicateImages3.csv")
dfObj.to_csv(r"DuplicateImages3.2.csv")