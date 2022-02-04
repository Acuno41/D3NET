import glob
import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
import time
from tensorflow.keras import backend as K


visualize = False
Path = r'test'
weightPath = r'drivableRegDatasetResults'
modelFileName = weightPath + r'/weights-improvement-56-0.04257.pb'

grid = 64
maxValue = 1024
scale = 1024/maxValue

K.set_learning_phase(0) # tespit suresini kisaltti.
output_names = ['dense_4/BiasAdd']
input_names = ['batch_normalization_1_input']

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


trt_graph = get_frozen_graph(modelFileName)

# Create session and load graph
tf_config = tf.ConfigProto() 
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')

# Get graph input size

for node in trt_graph.node:
    if 'batch_normalization_1_input' in node.name:
        size = node.attr['shape'].shape
        print(size)
        image_size = [size.dim[i].size for i in range(1, 4)]
        break
print("image_size: {}".format(image_size))

input_tensor_name = input_names[0] + ":0"
output_tensor_name = output_names[0] + ":0"

print("input_tensor_name: {}\noutput_tensor_name: {}".format(
    input_tensor_name, output_tensor_name))

output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)


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


        midPointsStack = []

        sTime = time.time()
        
        im = cv2.resize(im,(int(width/scale), int(height/scale)))
        height, width, channels = im.shape
        imagesStack = [np.expand_dims(np.zeros((maxValue, grid, 3)), axis=0)]*int(width/grid)

        im = (im[...,::-1].astype(np.float32)) / 255.0
        im = image.img_to_array(im)
        for i in range(1, int(width/grid+1)):

            gridLow = grid * (i - 1)
            gridHigh = grid * i
            gridIm = im[0:height, gridLow:gridHigh]
            
            midPoint = (gridLow + gridHigh) / 2
            midPointsStack.append(int(midPoint))

            #gridIm = image.img_to_array(gridIm)
            #gridIm = (gridIm[...,::-1].astype(np.float32)) / 255.0
            gridIm = np.expand_dims(gridIm, axis=0)
            imagesStack[i-1] = gridIm
        
        
        feed_dict = {
            input_tensor_name: np.vstack(imagesStack)
        }

        results = tf_sess.run(output_tensor, feed_dict)
        stopTime = time.time() - sTime
        if imgs == 0: continue
        totalTime += stopTime
        results = np.round(results * maxValue).tolist()

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
    print('Average Time=', round(totalTime/(len(imageFiles)-1),5))
    print('FPS=', (len(imageFiles)-1)/totalTime)

    tf_sess.close()