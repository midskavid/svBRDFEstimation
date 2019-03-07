import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import argparse
import random
import time
import os
import sys
import glob
import scipy
import scipy.misc


FOLDER_NAME = sys.argv[1]

print os.path.join(FOLDER_NAME,'*.jpg')
for file in glob.glob(os.path.join(FOLDER_NAME,'*.jpg')) : 
  print 'Converting ', file
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
  os.system("convert -resize 256x256! '"+ file +"' "+"'"+file+"'")


for file in glob.glob(os.path.join(FOLDER_NAME,'*.jpg')) : 
  prefix = file.split('.')[0]
  print 'Processing ', file
  img = cv.imread(file)
  #(H, W) = img.shape[:2]
  mask = np.zeros(img.shape[:2],np.uint8)
  bgdModel = np.zeros((1,65),np.float64)
  fgdModel = np.zeros((1,65),np.float64)
  # rect = (20,20,img.shape[0]-20,img.shape[1]-20)

  # play with mask...
  mask[:,:] = cv.GC_PR_FGD
  mask[img.shape[0]/2, img.shape[1]/2] =  cv.GC_FGD
  #print mask
  mask[:, img.shape[1]-50:] = cv.GC_BGD
  mask[:,0:50] = cv.GC_BGD

  # mask[img.shape[1]-30:,:] = cv.GC_BGD
  # mask[0:30,:] = cv.GC_BGD


  cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
  mask2 = np.where((mask==2)|(mask==0),0,255).astype('uint8')
  img = img*mask2[:,:,np.newaxis]
  cv.imwrite(prefix+'_mask'+'.jpg', mask2)
  #plt.imshow(mask2),plt.colorbar(),plt.show()


# img = cv.imread('/home/midi/Study/WI19/CSE291D/Project/Preprocess/IMG_20190217_142846778.jpg')
# mask = np.zeros(img.shape[:2],np.uint8)
# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
# rect = (20,20,img.shape[0]-20,img.shape[1]-20)

# # play with mask...
# mask[:,:] = cv.GC_PR_FGD
# mask[img.shape[0]/2, img.shape[1]/2] =  cv.GC_FGD
# #print mask
# mask[:, img.shape[1]-50:] = cv.GC_BGD
# mask[:,0:50] = cv.GC_BGD

# # mask[img.shape[1]-30:,:] = cv.GC_BGD
# # mask[0:30,:] = cv.GC_BGD


# cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
# mask2 = np.where((mask==2)|(mask==0),0,255).astype('uint8')
# img = img*mask2[:,:,np.newaxis]
# cv.imwrite('jojo.jpg',mask2)
# plt.imshow(mask2),plt.colorbar(),plt.show()


# img = cv.imread('/home/midi/Downloads/RealWorldImages/RealWorldImages/20190217_140216.jpg')
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(gray,0,255,cv.THRESH_TOZERO+cv.THRESH_OTSU)
# print thresh.shape
# cv.imshow('jojoj',thresh)
# cv.waitKey()
#plt.imshow(thresh), plt.show()