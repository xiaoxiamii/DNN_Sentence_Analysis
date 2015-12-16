#-*-uft-8-*-
"""
    Train a Convolutional Neural Networks on the short query sentiment classification task.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN, device=gpu, floatX=float32 python cnn_sen_clf.py

    Expect acc:

"""

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import sys
np.random.seed(1337) # for reproducibility

from keras.preprocessing import sequence
from keras.callbacks import Callback
from keras.optimizers import Adadelta
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils.np_utils import accuracy
from keras.utils import np_utils, generic_utils
from process_data import load_data
from keras.layers.recurrent import LSTM
from keras.layers.core import Reshape

#deep CNN
def deep_CNN(input_shape, label_class):
    #Two Convolutional Layers with Pooling Layer
    #config
    nb_filter = [250, 150, 50]
    nb_row = [1, 1]
    nb_col = [1, 1]
    nb_pool_size = [1, 1]

    #init
    model = Sequential()

    #First layers
    model.add(Convolution2D( nb_filter = 100 , nb_row = 3, nb_col = 300,
        border_mode = 'valid', activation = 'relu', 
        input_shape=(input_shape[0],input_shape[1],input_shape[2])
        ))
 
    model.add(Reshape(dims=(100*(input_shape[1]-2) ,300)))
    model.add(LSTM(300,return_sequences=True,input_length=100*(input_shape[1]-2)))
    
    model.add(Flatten())
    model.add(Dense(output_dim = 256, activation = 'relu',input_dim = 2048))
    model.add(Dropout(0.5))

    #Fully Connected Layer as output layer
    model.add(Dense(output_dim = label_class, activation='sigmoid',input_dim=256))
    model.add(Dropout(0.5))
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
    model.compile(loss='binary_crossentropy', class_mode = 'categorical',
            optimizer = adadelta)
    return model


if __name__ == "__main__":
    if(sys.argv[1] == ""):
        print ("argv[1] is the path of file made from process_data.py")
        exit()
    #config
    batch_size = 50
    nb_epoch = 25

    input_shape, w2v_dim, label_class, data, label = load_data(path =sys.argv[1],filter_h =5);
    label = np_utils.to_categorical(label, label_class)

    #print("Pad sequences(sample x time)")
    #data = sequence.pad_sequences(data)

    model = deep_CNN(input_shape, label_class)

    print("begin train model..")
    model.fit(data, label,
            batch_size = batch_size,
            nb_epoch = nb_epoch,
            shuffle = True,
            show_accuracy = True,
            verbose = 1,
            validation_split = 0.1
            )
            #validation_split = 0.1,

    """
    print("Validation...")
    val_loss, val_accuracy = model.evaluate(X_test, Y_test,
           show_accuracy = True, verbose = 0)
    print("val loss: %.4f \t val acc: %.4f"%(val_loss, val_accuracy))
    """



