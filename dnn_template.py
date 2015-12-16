#-*-uft-8-*-
"""
    Train a Deep Neural Networks on the short query sentiment classification task.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN, device=gpu, floatX=float32 python cnn_sen_clf.py

    Expect acc:

"""

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import sys
import time

#np.random.seed(1337) # for reproducibility

from keras.preprocessing import sequence
from keras.callbacks import Callback
from keras.optimizers import Adadelta, SGD, RMSprop, Adagrad, Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, TimeDistributedDense, Masking, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM, GRU
from keras.utils.np_utils import accuracy
from keras.utils import np_utils, generic_utils
from process_data import load_data, write_pre_loss
from keras.regularizers import l2

#model here
def DNN(data, input_shape, label_class):
    #config

    #init
    print("Begin build model..")
    model = Sequential()
    nb_samples, sentence_len, input_dim = data.shape
    print (data.shape)

    model.add( LSTM(output_dim = 300, input_dim=300))

    #Fully Connected Layer as output layer
    model.add(Dropout(0.5))
    model.add( Dense(output_dim=label_class, W_regularizer=l2(0.01)) )
    model.add(Activation("sigmoid"))
    #model.add(Dropout(0.5))


    #adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
    model.compile(loss='binary_crossentropy', 
            class_mode = 'binary',
            optimizer = 'rmsprop')

    return model


if __name__ == "__main__":
    if(sys.argv[1] == ""):
        print ("argv[1] is the path of file made from process_data.py")
        exit()
    #config
    seed = np.random.randint(0,1000)
    print (seed)
    #seed = 1337
    batch_size = 32
    nb_epoch = 25
    all_num = 10
    valid_rate = 0.9
    path = '../log/%s-dnn_log_test'%(time.strftime("%m-%d_%H-%M"))

    input_shape, w2v_dim, label_class, data, label = load_data(path
            =sys.argv[1],filter_h =5, model_type = "RNN");
    #label need to change
    label = np_utils.to_categorical(label, label_class)

    acces = np.zeros(all_num)
    for i in range(0, all_num):
        #random data
        np.random.seed(seed) 
        np.random.shuffle(data)
        np.random.seed(seed) 
        np.random.shuffle(label)

        #data split to train and test
        datasize = len(label)
        train_data = data[ : datasize*valid_rate]
        train_label = label[ : datasize*valid_rate]
        test_data = data[datasize*valid_rate: ]
        test_label = label[datasize*valid_rate:]

        print ("trainNum: %d"%(i))
        print ("train_file: %d"%(len(train_label)))
        print ("test_file: %d"%(len(test_label)))

        model = DNN(train_data, input_shape, label_class)

        print("Begin train model..")
        hist = model.fit(train_data, train_label, 
            batch_size = batch_size, 
            nb_epoch = nb_epoch,
            shuffle = True,
            show_accuracy = True,
            validation_data=(test_data,test_label),
            verbose = 1
            )

        write_pre_loss(hist, path, i, nb_epoch, seed)
        #count avg pre need to do
        """
        pre, recall =pre_recall_count(test_label, predict_label,label_class)
        for j in range(0, label_class):
            print ("Class:%d\tPre:%.4f\tRecall:%.4f"%(j,
                pre[j][1]/pre[j][0]),recall[j][1]/recall[j][0])
        """

    #print (("Mean Acc: %.4f")%np.mean(acces))
