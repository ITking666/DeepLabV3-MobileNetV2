# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 15:35:52 2018

@author: qiulin
"""

# import the necessary packages: skimage, imutils, cv2, numpy
from __future__ import print_function
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import imutils
import cv2
import numpy as np

def fruit_center_size(input, min_distance=4, noise_threshold=100, boxtype='circle'):
	image = input.copy()

	# convert the image to grayscale, then apply Otsu's thresholding
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	# find contours in the thresholded image
	_, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	# compute the exact Euclidean distance from every binary
	# pixel to the nearest zero pixel, then find peaks in this
	# distance map
	D = ndimage.distance_transform_edt(thresh)
	localMax = peak_local_max(D, indices=False, min_distance=min_distance,
							  labels=thresh)

	# perform a connected component analysis on the local peaks,
	# using 8-connectivity, then apply the Watershed algorithm
	markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
	labels = watershed(-D, markers, mask=thresh)

	centers = []
	fruitsizes = []
	circles = []
	rects = []
	# loop over the unique labels returned by the Watershed algorithm
	for label in np.unique(labels):
		# if the label is zero, we are examining the 'background'
		# so simply ignore it
		if label == 0:
			continue

		# otherwise, allocate memory for the label region and draw
		# it on the mask
		mask = np.zeros(gray.shape, dtype="uint8")
		mask[labels == label] = 255

		# detect contours in the mask and grab the largest one
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
								cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)

		# if the contour area is too small, we treat it as noise
		# so simply ignore it
		if int(cv2.contourArea(c))<noise_threshold:
			continue
		((x, y), r) = cv2.minEnclosingCircle(c)
		rects.append(cv2.minAreaRect(c))
		circles.append(((x, y), r))

	if boxtype == "circle":
		# consider overlapping suppression first
		i = 0
		j = 0
		while i < len(circles):
			while j < len(circles):
				if circles[i] == circles[j]:
					j += 1
					continue
				d_center = np.sqrt(
					(circles[i][0][0] - circles[j][0][0]) ** 2 + (circles[i][0][1] - circles[j][0][1]) ** 2)
				if d_center < circles[i][1] and d_center < circles[j][1]:
					r_new = 0.5 * (d_center + circles[i][1] + circles[j][1])
					x_0 = ((d_center + circles[i][1]) * circles[i][0][0] - circles[i][1] * circles[j][0][
						0]) / d_center
					y_0 = ((d_center + circles[i][1]) * circles[i][0][1] - circles[i][1] * circles[j][0][
						1]) / d_center
					x_new = r_new / circles[i][1] * (circles[i][0][0] - x_0) + x_0
					y_new = r_new / circles[i][1] * (circles[i][0][1] - y_0) + y_0
					circles[i] = None
					circles[j] = None
					i = 0
					j = 0
					circles.append(((x_new, y_new), r_new))
					while None in circles:
						circles.remove(None)
				j += 1
			i += 1
			j = 0
		# after suppression, draw the a circle enclosing the object and store the center and the size
		i = 0
		for circle in circles:
			x = circle[0][0]
			y = circle[0][1]
			r = circle[1]
			cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 1)
			cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
			centers.append((x, y))
			fruitsizes.append(np.pi * r * r)
			i = i + 1
	else:
		# draw a rectangle enclosing the object and store the center and the size
		i = 0
		for rect in rects:
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			centers.append(rect[0])
			fruitsizes.append(rect[1][0] * rect[1][1])
			cv2.drawContours(image, [box], 0, (0, 255, 0), 1)
			cv2.putText(image, "#{}".format(i + 1), (int(rect[0][0]) - 10, int(rect[0][1])),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
			i = i + 1
	print("[INFO] {} unique segments found".format(i+1))

	return image, centers, fruitsizes