import os
import cv2
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from scipy.misc import imsave

def preprocess(data_path):
    cross_labels, cross_data = [], []
    for dirname in os.listdir(data_path):
        if dirname=="train" or dirname=="valid":
            label, data = [], []
            full_p = os.path.join(data_path, dirname)
            for full_cl_p in glob.glob(os.path.join(full_p, "class*")):
                class_dir = os.path.basename(full_cl_p)
                pos = class_dir.find('_')
                l_num = class_dir[pos+1:]
                for image_path in glob.glob(os.path.join(full_cl_p, "*.png")):
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    img = np.expand_dims(img, axis=-1) / 255.
                    data.append(img)
                    label.append(l_num)
            cross_labels.append(label)
            cross_data.append(data)
        else:
            continue
    train, val, label_train, label_val = 0, 0, 0, 0
    for i in range(len(cross_data)):
        if len(cross_data[i])==50000:
            train = np.array(cross_data[i])
            label_train = np.array(cross_labels[i])
        elif len(cross_data[i])==10000:
            val = np.array(cross_data[i])
            label_val = np.array(cross_labels[i])
    return train, label_train, val, label_val

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

num_classes = 10
data_path = sys.argv[1]
x_train, train_labels, x_val, val_labels = preprocess(data_path)
y_train, y_val = to_categorical(train_labels, num_classes), to_categorical(val_labels, num_classes)
#print(x_train.shape, x_val.shape)
#print(y_train.shape, y_val.shape)

#build model
model, L_model, H_model = CNN()
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['accuracy'])

ckpt = ModelCheckpoint('CNN_model_e{epoch:02d}', # CNN_model_e{epoch:02d}_a{val_acc:.4f}
                       monitor='val_acc',
                       save_best_only=False,
                       save_weights_only=True,
                       verbose=1)
cb = [ckpt]
#training
epochs = 40
batch_size = 256
history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    callbacks=cb,
                    verbose=1)
#plot training curves
l = history.history['loss']
vl = history.history['val_loss']
acc = history.history['acc']
vacc = history.history['val_acc']

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(np.arange(epochs)+1, l, 'b', label='train loss')
plt.plot(np.arange(epochs)+1, vl, 'r', label='valid loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("loss curve")
plt.legend(loc='best')

plt.subplot(122)
plt.plot(np.arange(epochs)+1, acc, 'b', label='train accuracy')
plt.plot(np.arange(epochs)+1, vacc, 'r', label='valid accuracy')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("accuracy curve")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("training_curve.png")

