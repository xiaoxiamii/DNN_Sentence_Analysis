import numpy as np
from keras.utils import np_utils, generic_utils
from process_data import pre_recall_count
seed = 1337

data = np.empty(shape=(11,5))
label = np.empty(shape=(11))

for i in range(0,11):
    for j in range(0,5):
        data[i][j] = i+j;
    label[i] = i;

label = np_utils.to_categorical(label, 11)
print data
print label

np.random.seed(seed) # for reproducibility
np.random.shuffle(data)
np.random.seed(seed) # for reproducibility
np.random.shuffle(label)
data_size = len(label) 
train_data = data[:data_size*0.9]
train_label = label[:data_size*0.9]
test_data = data[data_size*0.9:]
test_label = label[data_size*0.9:]

predict_label = [1,2,3,4,5,6,7,8,9]
label_class = 11
pre, recall=pre_recall_count(train_label, predict_label,label_class)
for j in range(0, label_class):
    print ("Class:%d\tPre:%.4f\tRecall:%.4f"%(j,pre[j][1]/pre[j][0]),recall[j][1]/recall[j][0])


