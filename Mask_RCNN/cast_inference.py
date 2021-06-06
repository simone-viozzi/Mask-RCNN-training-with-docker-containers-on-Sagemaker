'''
Script for Mask_R-CNN training 
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
# TF DEBUG LEVELS: should be before tf import
#     0 = all messages are logged (default behavior)
#     1 = INFO messages are not printed
#     2 = INFO and W  ARNING messages are not printed
#     3 = INFO, WARNING, and ERROR messages are not printed

import cv2
import random
import imutils
import argparse
import numpy as np
from imutils import paths
from mrcnn import utils
from mrcnn import visualize
from mrcnn import model as modellib
from mrcnn.sagemaker_utils import *
from mrcnn.config import Config
from imgaug import augmenters as iaa
from PIL import Image
import base64
import zlib
import json
import io

# NOTE: used in the load_mask function
# don't move this declaration.
CLASS_NAMES = {
	1 : "chipping",
	2 : "deburring",
	3 : "holes",
	4 : "disk"
}

CLASS_COLOR = {
	1 : (0.0, 1.0, 1.0), # yellow
	2 : (0.0, 1.0, 0.0), # green
	3 : (0.0, 0.0, 1.0), # blue
	4 : (1.0, 0.0, 0.0)  # red
}

class castConfig(Config):
	"""
	Extension of Config class of the framework maskrcnn (mrcnn/config.py),
	"""

	def __init__(self, **kwargs):
		"""
		Overriding of same config variables
		and addition of others.
		"""
		self.__dict__.update(kwargs)
		super().__init__()

class castInferenceConfig(castConfig):
	
	NAME = "cast"
		
	# set the number of GPUs and images per GPU (which may be
	# different values than the ones used for training)
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# set the minimum detection confidence (used to prune out false
	# positive detections)
	DETECTION_MIN_CONFIDENCE = 0.6

	NUM_CLASSES = len(CLASS_NAMES) + 1

if __name__ == "__main__":

	ap = argparse.ArgumentParser()

	ap.add_argument("-i", "--image", help = "optional path to input image to segment" )
	
	args = vars(ap.parse_args())
	
	################################################################
	# DATASET TEST PATH
	# put here your path to the test dataset for inferencevisualization
	TEST_DATASET_PATH = ""
	################################################################

	################################################################
	# MODEL PATH definitions, 
	# put here your directoryes and your model name
	checkpoints_path = "/home/massi/Progetti/repository_simone/Mask-RCNN-training-with-docker-containers-on-Sagemaker/logs/cast_test_1/checkpoints/"
	MODEL = "mask_rcnn_cast_0039.h5"
	MODEL_PATH = os.path.sep.join([checkpoints_path, MODEL])
	################################################################

	# initialize the inference configuration
	config = castInferenceConfig()

	# initialize the Mask R-CNN model for inference
	model = modellib.MaskRCNN(mode="inference", config=config, checkpoints_dir=checkpoints_path)
	
	# load our trained Mask R-CNN
	model.load_weights(MODEL_PATH, by_name=True) # , exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
	
	files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.' + 'h5')]
	
	# load the input image, convert it from BGR to RGB channel
	# ordering, and resize the image
	image = cv2.imread(args["image"])
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = imutils.resize(image, width=1024)

	print("start inference...")

	# perform a forward pass of the network to obtain the results
	r = model.detect([image], verbose=1)[0]
	#print(r)

	# loop over of the detected object's bounding boxes and
	# masks, drawing each as we go along
	for i in range(0, r["rois"].shape[0]):
		mask = r["masks"][:, :, i]
		image = visualize.apply_mask(image, mask, CLASS_COLOR[r["class_ids"][i]], alpha=0.5)
		
		image = visualize.draw_box(image, r["rois"][i], CLASS_COLOR[r["class_ids"][i]])

	# convert the image back to BGR so we can use OpenCV's
	# drawing functions
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	# loop over the predicted scores and class labels
	for i in range(0, len(r["scores"])):
		# extract the bounding box information, class ID, label,
		# and predicted probability from the results
		(startY, startX, endY, end) = r["rois"][i]
		classID = r["class_ids"][i]
		label = CLASS_NAMES[classID]
		score = r["scores"][i]

		# draw the class label and score on the image
		text = "{}: {:.4f}".format(label, score)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

	# resize the image so it more easily fits on our screen
	image = imutils.resize(image, width=512)

	# show the output image
	cv2.imshow("Output", image)
	cv2.waitKey(0)
