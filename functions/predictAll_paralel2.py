import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import img_to_array
import cv2
import glob
import os
import time


visualize = False
#Path = r'/test'
Path = r'/val'
weightPath = r'/drivableRegDatasetResults'
modelFileName = weightPath + r'/weights-improvement-129-0.03991.hdf5'

K.set_learning_phase(0) # tespit suresini kisaltti.
grid = 64
maxValue = 1024
scale = 1024/maxValue
model = load_model(modelFileName)  
btchSize = 8

cities = glob.glob(Path + '/*')
print(cities)
for city in range(0, len(cities)):

    print("Curr File = {}".format(os.path.basename(cities[city])))
    outputs = Path + '/' + os.path.basename(cities[city]) + '/results'
    if not os.path.exists(outputs):
        os.mkdir(outputs)
    totalTime = 0
    imageFiles = glob.glob(cities[city] + '/*png')
    for imgs in range(0, len(imageFiles)):
        txtFile = open(outputs + '/' + os.path.basename(imageFiles[imgs])[0:-4] + '.txt', 'w')
        im = cv2.imread(imageFiles[imgs])
        height, width, channels = im.shape


        sTime = time.time()
        im = cv2.resize(im,(int(width/scale), int(height/scale)))
        
        height, width, channels = im.shape
        imagesStack = [np.expand_dims(np.zeros((maxValue, grid, 3)), axis=0)]*int(width/grid)
        midPointsStack = []

        
        im = (im[...,::-1].astype(np.float32)) / 255.0
        for i in range(1, int(width/grid+1)):

            gridLow = grid * (i - 1)
            gridHigh = grid * i
            gridIm = im[0:height, gridLow:gridHigh]
            
            midPoint = (gridLow + gridHigh) / 2
            midPointsStack.append(int(midPoint))

            gridIm = img_to_array(gridIm)
            #gridIm = (gridIm[...,::-1].astype(np.float32)) / 255.0
            gridIm = np.expand_dims(gridIm, axis=0)
            imagesStack[i-1] = gridIm

        imagesStack = np.vstack(imagesStack)
        results = model.predict(imagesStack,batch_size=btchSize)
        stopTime = time.time() - sTime
        if imgs == 0: continue
        totalTime += stopTime
        results = np.round(results * maxValue).tolist()
        #print(midPointsStack)
        #print(results)
        tempIm = im.copy()
        if visualize:
            oldPoint = (0, maxValue)
            Point = (0, maxValue)
            for i in range(0, int(width/grid+1)-1):

                Point = (midPointsStack[i], int(results[i][0]))

                cv2.line(tempIm, oldPoint, Point, (255, 0, 0), 3)
                cv2.circle(tempIm, Point, 3, (0, 0, 255), -1)
                oldPoint = Point

                txtFile.write(str(Point)+'\n')
            txtFile.close()
            cv2.line(tempIm, Point, (2048, 1024), (255, 0, 0), 3)
            cv2.imwrite(outputs + '/' + os.path.basename(imageFiles[imgs]), tempIm)
    print('Average Time=', totalTime/(len(imageFiles)-1))
    print('FPS=',(len(imageFiles)-1)/totalTime)