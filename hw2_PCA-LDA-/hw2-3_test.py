import os
import sys
import glob
import cv2
import csv
import numpy as np
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical

def preprocess(data_path):
    data, label, f_name = [], [], []
    for image_path in glob.glob(os.path.join(data_path, "*.png")):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=-1) / 255.
        data.append(img)
        name = os.path.basename(image_path)
        l_num = int(name[0])
        label.append(l_num)
        f_name.append(name[:4])
    return np.array(data), np.array(label), f_name

def CNN():
    img_input = Input(shape=(28, 28, 1))
    c1 = Conv2D(6, (5, 5), padding='valid', activation='relu', name='c1')(img_input)
    s1 = MaxPooling2D(pool_size=(2,2), name='s1')(c1)
    c2 = Conv2D(16, (5, 5), padding='valid', activation='relu', name='c2')(s1)
    s2 = MaxPooling2D(pool_size=(2,2), name='s2')(c2)
    flat = Flatten()(s2)
    fc1 = Dense(120, activation='relu', name='fc1')(flat)
    fc2 = Dense(84, activation='relu', name='fc2')(fc1)
    output = Dense(num_classes, activation='softmax', name='output')(fc2)

    model = Model(img_input, output)
    L_model = Model(img_input, c1)
    H_model = Model(img_input, s2)
    return model, L_model, H_model

data_path, pred_csv = sys.argv[1], sys.argv[2]
num_classes = 10
model_name = 'CNN_model_e40'
model, _, _ = CNN()
model.load_weights(model_name)
x_test, labels, f_name = preprocess(data_path)
y_test = to_categorical(labels, num_classes)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['accuracy'])
prediction = model.predict(x_test)
argmax_lst = np.argmax(prediction, axis=1)
i=0
with open(pred_csv, 'w') as csvfile:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for x in argmax_lst:
        writer.writerow({'id': f_name[i], 'label': str(x)})
        i+=1
#accuracy = model.evaluate(x_test, y_test, verbose=0)
#print('Testing accuracy: {}'.format(accuracy))