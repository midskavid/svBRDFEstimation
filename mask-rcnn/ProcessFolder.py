import numpy as np
import argparse
import random
import time
import cv2
import os
import sys
import glob
import scipy
import scipy.misc


FOLDER_NAME = sys.argv[1]

#print os.path.join(FOLDER_NAME,'*.jpg')
for file in glob.glob(os.path.join(FOLDER_NAME,'*.jpg')) : 
  print 'Processing ', file
  img = scipy.misc.imread(file)
  # if img.shape[0] < img.shape[1] : 
  #   #landscape.. crop 
  #   height = img.shape[0]
  #   width = img.shape[1]
  #   newImg = img[:,width/2-height/2:width/2-height/2+height,:]
  #   newImg = scipy.misc.toimage(newImg)
  #   newImg.save(file)

  # else :
  #   height = img.shape[0]
  #   width = img.shape[1]
  #   newImg = img[height/2-width/2:height/2-width/2+width,:,:]
  #   newImg = scipy.misc.toimage(newImg)
  #   newImg.save(file)
  os.system('convert -resize 256x256! '+ file +' '+file)


print 'Loading network..'
weightsPath = 'mask-rcnn-coco/frozen_inference_graph.pb'
configPath = 'mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
print 'Loaded...'

for file in glob.glob(os.path.join(FOLDER_NAME,'*.jpg')) : 
  prefix = file.split('.')[0]
  print 'Processing ', file
  image = cv2.imread(file)
  (H, W) = image.shape[:2]
  blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
  net.setInput(blob)
  start = time.time()
  (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

  for i in range(0, boxes.shape[2]):
    # extract the class ID of the detection along with the confidence
    # (i.e., probability) associated with the prediction
    classID = int(boxes[0, 0, i, 1])
    confidence = boxes[0, 0, i, 2]

    # filter out weak predictions by ensuring the detected probability
    # is greater than the minimum probability
    if confidence > 0.3:

      # scale the bounding box coordinates back relative to the
      # size of the image and then compute the width and the height
      # of the bounding box
      clone = image[:,:,0]
      clone[:,:] = 0
      box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
      (startX, startY, endX, endY) = box.astype("int")
      boxW = endX - startX
      boxH = endY - startY

      mask = masks[i, classID]
      mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
      mask = (mask > 0.3)
      clone[startY:endY, startX:endX][mask] = 255
      cv2.imwrite(prefix+'_mask'+str(i)+'.jpg', clone)
      #cv2.imwrite(prefix+'_mask00'+str(i)+'.jpg', instance)

