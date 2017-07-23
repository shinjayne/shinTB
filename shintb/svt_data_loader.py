import xml.etree.ElementTree as ET
import cv2
import numpy as np
import random
#All feeded data should be in [x,y,w,h] format(left top based corner box)
class SVTDataLoader:
	def __init__(self, trainPath=None, testPath=None):
		print("SVT Data Loader instance Initialing...")

		trainList = None
		testList = None
		if trainPath:
			self.trainList = self.parseTree(trainPath)
		if testPath:
			self.testList = self.parseTree(testPath)

		print("Complete !!!")
	# parseTree
	# input : xml file path
	# output : dataset >> list of (name, rectangle)

	def parseTree(self, path):
		dataset = []
		# dataset : list of (name, rectangle)
		# name : image name String
		# rectangle : list of groundtruth boxes ([x,y,w,h] , 0)
		# [x,y,w,h] : scaled (devided by 300)
		tree = ET.parse(path)
		root = tree.getroot()
		for image in root.findall('image'):
			name = image.find('imageName').text
			rectangles = []
			taggedRectangles = image.find('taggedRectangles')
			for rectangle in taggedRectangles.findall('taggedRectangle'):
				h = float(rectangle.get('height')) / 300.0
				w = float(rectangle.get('width')) / 300.0
				x = float(rectangle.get('x')) / 300.0
				y = float(rectangle.get('y')) / 300.0
				rectangles.append(([x,y,w,h], 0))
			dataset.append((name, rectangles))
		return dataset


	# nextBatch
	# input : batch_size
	# output : images, ann

	# generate new random batch dataset which has length of 'batch_size' from 'datalist'

	def nextBatch(self, batches, dataset='train'):
		imgH = 300
		imgW = 300
		if dataset == 'train':
			datalist = self.trainList
		if dataset == 'test':
			datalist = self.testList
		randomIndex = random.sample(range(len(datalist)), batches)
		images = []
		gtboxes =[]

		# datalist : list of (name, rectangle)
		# name : image name String
		# rectangle : list of groundtruth boxes ([x,y,w,h] , 0)

		# images : list of <image numpy array>
		# gtboxes : list of <rectangle = list of gt_boxes >
		#  images[i] `s groundtruth boxs => gtboxes[i]
		for index in randomIndex:
			fileName = './svt1/' + datalist[index][0]
			img = cv2.imread(fileName, cv2.IMREAD_COLOR)
			resized = cv2.resize(img, (imgW, imgH))
			resized = np.multiply(resized, 1.0/255.0)
			images.append(resized)
			gtboxes.append(datalist[index][1])
		images = np.asarray(images)
		return images, gtboxes

