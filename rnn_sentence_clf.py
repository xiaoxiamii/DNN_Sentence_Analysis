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
from keras.optimizers import Adadelta, SGD, RMSprop, Adagrad
from keras.models import Sequential
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

    #init
    model = Sequential()


    #model.add(Dropout(0.25,input_shape = (input_shape[1], input_shape[2] )))
    #RNN layers
    #model.add( LSTM(output_dim = 128,input_dim = 300, return_sequences=False) )

    #model.add( Masking(mask_value = 0))
    print (data.shape)
    nb_samples, timesteps, input_dim = data.shape
    nb_hidden_nodes = timesteps
    print (timesteps)
    #model.add( LSTM(nb_hidden_nodes, activation='sigmoid', inner_activation='hard_sigmoid', input_dim=input_dim ))
    model.add( LSTM(output_dim = 128, inner_activation='hard_sigmoid',input_dim=300 ))
    #model.add( LSTM(output_dim = 128, activation='sigmoid',input_dim=300 ))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=label_class))
    model.add(Activation("softmax"))
    #model.add(Dropout(0.5))

    #Fully Connected Layer as output layer

    #model.add(Flatten())
    #model.add(Dense(output_dim = label_class, activation='sigmoid'));
    #adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
    model.compile(loss='binary_crossentropy', class_mode = 'binary',
            optimizer = 'rmsprop')

    return model


if __name__ == "__main__":
    if(sys.argv[1] == ""):
        print ("argv[1] is the path of file made from process_data.py")
        exit()
    #config
    batch_size = 32
    nb_epoch = 30
    all_num = 10
    valid_rate = 0.9

    input_shape, w2v_dim, label_class, data, label = load_data(path
            =sys.argv[1],filter_h =5, model_type = "RNN");
    label = np_utils.to_categorical(label, label_class)

    #data = sequence.pad_sequences(data)

    #model = parallel_CNN(input_shape, label_class)

    print("begin train model..")
    acces = np.zeros(all_num)
    for i in range(0, all_num):
        #random data
        seed = np.random.randint(0,1000)
        print("Seed:%d"%(seed))
        np.random.seed(seed) # for reproducibility
        np.random.shuffle(data)
        np.random.seed(seed) # for reproducibility
        np.random.shuffle(label)


        datasize = len(label)
        train_data = data[ : datasize*valid_rate]
        train_label = label[ : datasize*valid_rate]
        test_data = data[datasize*valid_rate: ]
        test_label = label[datasize*valid_rate:]

        #print("Pad sequences(sample x time)")
        print ("trainNum: %d"%(i))
        print ("train_file: %d"%(len(train_label)))
        print ("test_file: %d"%(len(test_label)))

        model = deep_RNN(train_data, input_shape, label_class)
        model.fit(train_data, train_label, 
            batch_size = batch_size, nb_epoch = nb_epoch,
            shuffle = True,
            show_accuracy = True,
            validation_data=(test_data,test_label),
            verbose = 1
            )
        hist = model.evaluate(test_data, test_label, show_accuracy=True,verbose=0)
        print (hist)
        #access[i] = hist.history['acc'][0]
        #print(access[i])
        predict_label = model.predict_classes(test_data)
        #print (predict_label)
        """
        pre, recall =pre_recall_count(test_label, predict_label,label_class)
        for j in range(0, label_class):
            print ("Class:%d\tPre:%.4f\tRecall:%.4f"%(j,
                pre[j][1]/pre[j][0]),recall[j][1]/recall[j][0])
        """

    print (("Mean Acc: %.4f")%np.mean(acces))

    """
    print("Validation...")
    val_loss, val_accuracy = model.evaluate(X_test, Y_test,
           show_accuracy = True, verbose = 0)
    print("val loss: %.4f \t val acc: %.4f"%(val_loss, val_accuracy))
    """



