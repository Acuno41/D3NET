# import the necessary packages
import pandas as pd
import numpy as np
import glob
import cv2
import os


def load_road_images(df, inputPath):
	# initialize our images array (i.e., the house images themselves)
	images = []

	# loop over the indexes of the houses
	for i in df.index.values:

		basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])
		housePaths = sorted(list(glob.glob(basePath)))

		inputImages = []
		outputImage = np.zeros((1024, 64, 3), dtype="uint8")
		print(i)
		# loop over the input house paths
		for housePath in housePaths:
			# update the list of input images
			image = cv2.imread(housePath)
			inputImages.append(image)

		outputImage[0:1024, 0:64] = inputImages[0]

		# cv2.imshow("output", outputImage)
		# cv2.waitKey(0)

		# add the tiled image to our set of images the network will be
		# trained on
		images.append(outputImage)

	# return our set of images
	return np.array(images)


def load_road_attributes(inputPath):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	cols = ["Name", "Border"]
	df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
	return df

