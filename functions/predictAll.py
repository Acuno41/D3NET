import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from keras import backend as K
import cv2
import glob
import os
import time


Path = r'/Desktop/test'
outputPath = r'/Desktop/test'

modelFileName = r'/drivableRegDatasetResults/weights-improvement-185-0.00844.hdf5'
K.set_learning_phase(0) # tespit suresini kisaltti.
grid = 64
maxValue = 1024
scale = 1024/maxValue
model = load_model(modelFileName)  


totalTime = 0
cities = glob.glob(Path + '/*')
print(cities)
for city in range(0, len(cities)):

    print("Curr File = {}".format(os.path.basename(cities[city])))
    outputs = outputPath + '/' + os.path.basename(cities[city])
    if not os.path.exists(outputs):
        os.mkdir(outputs)
    #sTime = time.time()
    
    imageFiles = glob.glob(cities[city] + '/*png')
    for imgs in range(0, len(imageFiles)):
        
        #txtFile = open(outputs + '\\' + os.path.basename(imageFiles[imgs])[0:-4] + '.txt', 'w')
        im = cv2.imread(imageFiles[imgs])
        height, width, channels = im.shape
        im = cv2.resize(im,(int(width/scale), int(height/scale)))
        tempIm = im.copy()
        im = im / 255.0
        height, width, channels = im.shape
        oldPoint = (0, maxValue)
        Point = (0, maxValue)
        sTime = time.time()
        for i in range(1, int(width/grid+1)):

            gridLow = grid * (i - 1)
            gridHigh = grid * i
            gridIm = im[0:height, gridLow:gridHigh]
            gridIm = np.expand_dims(gridIm, axis=0)

            result = model.predict(gridIm)
            midPoint = (gridLow + gridHigh) / 2
            Point = (int(midPoint), int(result * maxValue))

            cv2.line(tempIm, oldPoint, Point, (255, 0, 0), 3)
            cv2.circle(tempIm, Point, 3, (0, 0, 255), -1)
            oldPoint = Point

            #txtFile.write(str(Point)+'\n')
        #txtFile.close()
        cv2.line(tempIm, Point, (2048, 1024), (255, 0, 0), 3)
        cv2.imwrite(outputs + '\\' + os.path.basename(imageFiles[imgs]), tempIm)
        stopTime = time.time() - sTime
        if imgs == 0:  continue
        totalTime = totalTime + stopTime
        
    print('Elapsed Time = ', totalTime/len(imageFiles))
        


