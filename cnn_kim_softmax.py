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

#model here
def DNN(data, input_shape, label_class):
    #conf
    input_h = input_shape[1]
    input_w = input_shape[2]
    print ("%d, %d, %d" %(input_shape[0], input_shape[1], input_shape[2]))

    filter_w = input_w
    filter_hs = [3,4,5]

    hidden_units = [100,2]
    dropout_rate = [0.5]
    lr_decay = 0.95
    activation = 'relu'
    feature_maps = hidden_units[0]
    pool_shapes = []
    filter_shapes = []
    for filter_h in filter_hs:
        filter_shapes.append( (feature_maps, filter_h, filter_w))
        pool_shapes.append((input_h - filter_h +1, input_w - filter_w+1))
    sub_models = []
    print (pool_shapes)
    print (filter_shapes)
    for i in range( len(pool_shapes) ):
        pool_shape = pool_shapes[i]
        filter_shape = filter_shapes[i]
        sub_model = Sequential()
        sub_model.add( Convolution2D(nb_filter = 100, nb_row = filter_shape[1], nb_col = filter_shape[2], 
                border_mode='valid', activation='relu',
                input_shape=(input_shape[0], input_shape[1], input_shape[2])
                ))
        sub_model.add( MaxPooling2D(pool_size=(pool_shape[0], pool_shape[1])) )
        sub_models.append(sub_model)

    model = Sequential()
    model.add(Merge(sub_models, mode='concat'))

    #Fully Connected Layer with dropout

    model.add(Flatten())
    model.add( Dense(output_dim=label_class, activation='softmax'))
    model.add(Dropout(0.5))

    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
    model.compile(loss='binary_crossentropy', class_mode = 'binary',optimizer=adadelta)

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
    path = './log/cnn_kim_softmax_log'

    input_shape, w2v_dim, label_class, data, label = load_data(path
            =sys.argv[1],filter_h =5, model_type = "CNN");
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
        hist = model.fit([train_data, train_data, train_data], train_label, 
            batch_size = batch_size, 
            nb_epoch = nb_epoch,
            shuffle = True,
            show_accuracy = True,
            validation_data=([test_data, test_data, test_data],test_label),
            verbose = 1
            )

        write_pre_loss(hist, path, i, nb_epoch,seed)
        #count avg pre need to do
        """
        pre, recall =pre_recall_count(test_label, predict_label,label_class)
        for j in range(0, label_class):
            print ("Class:%d\tPre:%.4f\tRecall:%.4f"%(j,
                pre[j][1]/pre[j][0]),recall[j][1]/recall[j][0])
        """

    #print (("Mean Acc: %.4f")%np.mean(acces))
