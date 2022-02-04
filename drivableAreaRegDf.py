# USAGE
# python drivableAreaRegDf.py --dataset "SampleTrainDataset"

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks
from functions import datasets
from functions import models
import argparse
import matplotlib.pyplot as plt
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True, help="path to input dataset of images")
args = vars(ap.parse_args())

# load dataframe GT's
TrainImagesPath = args["dataset"]+"/train"
ValImagesPath = args["dataset"] + "/val"
print("[INFO] loading dataset attributes...")
inputPathTrain = os.path.sep.join([args["dataset"], "drivableDatasetTrain.txt"])
dfTrain = datasets.load_road_attributes(inputPathTrain)
inputPathVal = os.path.sep.join([args["dataset"], "drivableDatasetVal.txt"])
dfVal = datasets.load_road_attributes(inputPathVal)

# Normalize the GT's
maxValue = 1024 # image height
gridWidth = 64  # grid image width

dfTrain["Border"] = dfTrain["Border"]/maxValue
dfVal["Border"] = dfVal["Border"]/maxValue
print("[INFO] loading Images...")
# Generate Batch image data and normalize
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_dataframe(dataframe=dfTrain, directory=TrainImagesPath, x_col="Name", y_col="Border",
                                            class_mode="raw", target_size=(maxValue, gridWidth), batch_size=16)
valid_generator = datagen.flow_from_dataframe(dataframe=dfVal, directory=ValImagesPath, x_col="Name", y_col="Border",
                                            class_mode="raw", target_size=(maxValue, gridWidth), batch_size=16)

# load model and parameters
#model = models.cnn_network(maxValue, gridWidth, 3, regress=True)
model = models.cnn_network(maxValue, gridWidth, 3, regress=True)
model.summary()
opt = Adam(lr=1e-4, decay=1e-4 / 200)
model.compile(loss="mean_absolute_error", optimizer=opt)
# "mean_squared_error"
# check point save path

filepath = r"TrainResults/weights-improvement-{epoch:02d}-{val_loss:.5f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

# train the model
print("[INFO] training model...")

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                #    shuffle=True,
                    callbacks=callbacks_list,
                    epochs=200)

# summarize history for loss
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
