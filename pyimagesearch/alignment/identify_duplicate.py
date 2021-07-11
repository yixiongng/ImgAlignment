from __future__ import division

import cv2
import numpy as np
import glob
import pandas as pd
import timeit

start = timeit.default_timer()
listOfTitles1 = []
listOfTitles2 = []
listOfSimilarities = []

# orb and brute force 
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher()

# Load all the images1
countInner = 0
countOuter = 1

folder = "./scans/picture/*"

siftOut = {}
for a in glob.iglob(folder,recursive=True):
    if not a.lower().endswith(('.jpg','.png','.tif','.tiff','.gif')):
        continue

    image1 = cv2.imread(a)

    kp_1, desc_1 = orb.detectAndCompute(image1, None)
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

        (kp_2,desc_2) = siftOut[b]
        matches = bf.knnMatch(desc_1, desc_2, k=2)
        goods = []

        for  i,pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < 0.6 *n.distance:
                    goods.append(m)
            except ValueError:
                pass
        
        number_keypoints = 0
        if len(kp_1) >= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        percentage_similarity = float(len(goods)) / number_keypoints * 100

        listOfSimilarities.append(str(int(percentage_similarity)))
        listOfTitles2.append(b)
        listOfTitles1.append(a)

    countInner = 0
    if a.lower().endswith(('.jpg','.png','.tif','.tiff','.gif')):
        countOuter += 1

zippedList =  list(zip(listOfTitles1,listOfTitles2, listOfSimilarities))

print(zippedList)

dfObj = pd.DataFrame(zippedList, columns = ['Original', 'Title' , 'Similarity'])

dfObj.to_csv(r"DuplicateImages3.2.csv")
stop = timeit.default_timer()

print('Time: ', stop - start) 