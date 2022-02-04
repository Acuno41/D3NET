import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import time
from tensorflow.keras import backend as K

K.set_learning_phase(0) # tespit suresini kisaltti.
output_names = ['dense_4/BiasAdd']
input_names = ['batch_normalization_1_input']

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


trt_graph = get_frozen_graph('./drivableModel/weights-improvement-129-0.03991.pb')

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

imagePath = './drivableModel/berlin_000000_000019_leftImg8bit.png'
grid = 64
maxValue = 1024
scale = 1024/maxValue
totalTime = 0
totalParcala = 0
totalPredict = 0
for j in range(41):
    im = cv2.imread(imagePath)
    height, width, channels = im.shape
    im = cv2.resize(im,(int(width/scale), int(height/scale)))
    tempIm = im.copy()
    
    
    height, width, channels = im.shape
    imagesStack = [np.expand_dims(np.zeros((maxValue, grid, 3)), axis=0)]*int(width/grid)
    midPointsStack = []
    sTime = time.time()
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
    stopTime2 = time.time() - sTime
    sTime2 = time.time()
    one_prediction = tf_sess.run(output_tensor, feed_dict)
    
    stopTime = time.time() - sTime
    stopTime3 = time.time() - sTime2
    if j== 0: continue
    totalTime += stopTime
    totalParcala += stopTime2
    totalPredict += stopTime3
print('Average Time= ', round(totalTime/(41-1),5))
print('Average Time Parcala = ', round(totalParcala/(41-1),5))
print('Average Time Predict = ', round(totalPredict/(41-1),5))
print('FPS = ', 40/totalTime)
tf_sess.close()
