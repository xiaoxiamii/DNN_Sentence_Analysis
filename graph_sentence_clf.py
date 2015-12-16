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
from keras.optimizers import Adadelta, SGD, RMSprop, Adagrad, Adam
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, TimeDistributedDense, Masking
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM, GRU
from keras.utils.np_utils import accuracy
from keras.utils import np_utils, generic_utils
from process_data import load_data

#deep RNN
def deep_RNN(data, input_shape, label_class):
    #config
    nb_filter = [250, 150, 50]
    nb_row = [1, 1]
    nb_col = [1, 1]
    nb_pool_size = [1, 1]

    print (data.shape)
    nb_samples, maxlen, dim = data.shape
    data = np.array(data)

    #init
    print ('build model')
    model = Graph()
    model.add_input(name='input', input_shape=(maxlen,dim))
    model.add_node(LSTM(64), name='forward',input='input');
    #model.add_node(LSTM(64, go_backwards=True), name='backward',input='input')
    model.add_node(LSTM(64), name='backward',input='input')
    model.add_node(Dropout(0.5), name='dropout', inputs=['forward','backward'])
    model.add_node(Dense(1, activation='sigmoid'), name='sigmoid',input='dropout')
    model.add_output(name='output', input='sigmoid')
    model.compile('adam', {'output': 'binary_crossentropy'})

    #adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)

    return model


if __name__ == "__main__":
    if(sys.argv[1] == ""):
        print ("argv[1] is the path of file made from process_data.py")
        exit()
    #config
    batch_size = 16
    nb_epoch = 30
    all_num = 10

    input_shape, w2v_dim, label_class, data, label = load_data(path
            =sys.argv[1],filter_h =5, model_type = "RNN");
    label = np_utils.to_categorical(label, label_class)

    #print("Pad sequences(sample x time)")
    #data = sequence.pad_sequences(data)

    model = deep_RNN(data, input_shape, label_class)
    #model = parallel_CNN(input_shape, label_class)

    print("begin train model..")
    acces = np.zeros(all_num)
    for i in range(0, all_num):
        print (data.shape)
        print ("trainNum: %d"%(i))
        hist =model.fit({'input':data,'output':label}, 
            batch_size = batch_size,
            nb_epoch = 4,
            )
        acces[i] = hist.history['acc'][0]

    print (("Mean Acc: %.4f")%np.mean(acces))

    """
    print("Validation...")
    val_loss, val_accuracy = model.evaluate(X_test, Y_test,
           show_accuracy = True, verbose = 0)
    print("val loss: %.4f \t val acc: %.4f"%(val_loss, val_accuracy))
    """



