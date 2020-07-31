
!pip install tensorflow==1.13.1
import tensorflow as tf
print(tf.__version__)

"""---"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from PIL import Image
import numpy as np
import os

seed = 7
np.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')

print(X_train.shape)


def load_images_to_data(image_label, image_directory, features_data, label_data):
    list_of_files = os.listdir(image_directory)
    print(image_directory)
    count=0
    for file in list_of_files:
        count+=1
        print(count)
        image_file_name = os.path.join(image_directory, file)
        if ".jpg" in image_file_name:
            img = Image.open(image_file_name).convert("L")
            img = np.resize(img, (28,28,1))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1,28,28,1)
            features_data = np.append(features_data, im2arr, axis=0)
            label_data = np.append(label_data, [image_label], axis=0)
    return features_data, label_data

X_train, y_train = load_images_to_data('10', '/content/drive/My Drive/operator/updated_operator/+', X_train, y_train)
print("finished")
X_test, y_test = load_images_to_data('10', '/content/drive/My Drive/operator/updated_operator/validation/+', X_test, y_test)
print("plus")

X_train, y_train = load_images_to_data('11', '/content/drive/My Drive/operator/updated_operator/-', X_train, y_train)
print("finished")
X_test, y_test = load_images_to_data('11', '/content/drive/My Drive/operator/updated_operator/validation/-', X_test, y_test)
print("sub")
X_train, y_train = load_images_to_data('12', '/content/drive/My Drive/operator/updated_operator/X', X_train, y_train)
print("finished")
X_test, y_test = load_images_to_data('12', '/content/drive/My Drive/operator/updated_operator/validation/X', X_test, y_test)
print("mul")
X_train, y_train = load_images_to_data('13', '/content/drive/My Drive/operator/updated_operator/div', X_train, y_train)
print("finished")
X_test, y_test = load_images_to_data('13', '/content/drive/My Drive/operator/updated_operator/validation/div', X_test, y_test)
print("div")

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

X_train/=255
X_test/=255

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')
print(X_train.shape)


number_of_classes = 14
y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)


model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(X_train.shape[1],X_train.shape[2], 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=7, batch_size=200)

model.save('Operator.h5')

metrics = model.evaluate(X_test, y_test, verbose=0)
print("Metrics(Test loss & Test Accuracy): ")
print(metrics)

import cv2
import numpy as np
img = Image.open('/content/test.jpg').convert("L")
img=np.invert(img)
cv2.imwrite("image.jpg",np.array(img))
img = np.resize(img, (28,28,1))
im2arr = np.array(img)
im2arr = im2arr.reshape(1,28,28,1)
y_pred = model.predict_classes(im2arr)
print(y_pred)

from keras import backend as K

K.set_learning_phase(0)

from keras.models import load_model
model = load_model('OperatorCNN.h5')
print(model.outputs)
# [<tf.Tensor 'dense_2/Softmax:0' shape=(?, 10) dtype=float32>]
print(model.inputs)

from keras import backend as K
import tensorflow as tf

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

tf.train.write_graph(frozen_graph, "model", "Operator_model.pb", as_text=False)

import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
  f = gfile.FastGFile("./data.pb", 'rb')
graph_def = tf.GraphDef()

graph_def.ParseFromString(f.read())
f.close()

sess.graph.as_default()

tf.import_graph_def(graph_def)
