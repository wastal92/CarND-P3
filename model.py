import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Conv2D, Activation, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.image as mpimg

lines = []
images = []
measurements = []
correction = 0.25
#Read the driving_log
with open('./driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

#Read the images and steering angles
for line in lines:
    for i in range(3):
        source_path = line[i]
        file_name = source_path.split("\\")[-1]
        current_path = './IMG/' + file_name
        image = mpimg.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        #Use center, left and right images
        if i==0:
            measurements.append(measurement)
        elif i==1:
            measurements.append(measurement + correction)
        elif i==2:
            measurements.append(measurement - correction)

X_train = np.array(images)
y_train = np.array(measurements)

#Split the train set
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size= 0.2)

#Define generator function
def generator(X, y, batch_size = 32):
    num_samples = len(y)
    while 1:
        X, y = shuffle(X, y)
        for offset in range(0, num_samples, batch_size):
            X_batch = X[offset:offset+batch_size]
            y_batch = y[offset:offset+batch_size]
            yield shuffle(X_batch, y_batch)

#Biuld the model
model = Sequential()
model.add(Lambda(lambda x: x /225.0 -0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
model.add(Conv2D(24, 5, strides = (2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(36, 5, strides = (2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Conv2D(48, 5, strides = (2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Conv2D(64, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Conv2D(64, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

batch_size = 64
train_generator = generator(X_train, y_train, batch_size=batch_size)
valid_generator = generator(X_valid, y_valid, batch_size=batch_size)

#Train model
model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator, steps_per_epoch=int(len(y_train)/batch_size)+1,
                    epochs = 40, validation_data=valid_generator, validation_steps=
                    int(len(y_valid)/batch_size)+1, verbose=1)
model.save('model.h5')