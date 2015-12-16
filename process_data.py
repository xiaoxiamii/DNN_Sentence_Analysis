#-*- utf-8 -*-
"""
input: label data, wordvector data
output:
    revs: list.     Store the all informaiton of sentences: label, text, num_words
    vocab: dict: word, nums.    the dict to store all appeared word in train_test data
    w2v: dict: word, word_vector. 
    word_idx_map: dict: word, word_id.
    W: dict: word_id, word_vector. part of vector from Google 300 vector
    W2: dict: word_id, word_vector. All random vector, in range(-0.25, 0.25)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))

"""
import numpy as np
np.random.seed(1337) # for reproducibility

import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import gc
import time


#data_folder: the path to load data
#clean_string : True = clean the sentence
def build_data_cv(data_folder, clean_string=True):
    """
    Loads sentence data ;
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"label":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split())
                      }
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"label":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split())
                      }
            revs.append(datum)
    return revs, vocab
    
def get_W(word_vecs, w2v_dim=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, w2v_dim))            
    W[0] = np.zeros(w2v_dim)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, w2v_dim=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    cnt = 0.0;
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, w2v_dim)  
            cnt +=1
    return cnt

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def load_data(path, word_vectors="-word2vec", filter_h=5, model_type="CNN"):

    print "Loading data..."
    X = cPickle.load(open(path, "rb"))
    #cPickle.dump([max_l, w2v_dim, revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    max_len, w2v_dim, revs, W, W2, word_idx_map, vocab = X[0], X[1], X[2],X[3], X[4], X[5], X[6]

    if (word_vectors == "-word2vec"):
        w2v = W
        del W2
    elif (word_vectors == "-rand"):
        w2v = W2
        del W

    del X
    revs_size = len(revs)
    vocab_size = len(vocab)
    pad = filter_h -1
    w2v_size = len(w2v) 
    #gc.collect()
    print "data loaded"
    print "number of sentences: " + str(revs_size)
    print "vocab size: " + str(vocab_size)
    print "max sentence length: " + str(max_len)
    print "num words already in word2vec: " + str(w2v_size)


    #input_shape
    input_shape = []
    input_shape.append(1)
    input_shape.append(max_len +  2*pad)
    input_shape.append(w2v_dim)

    w2v[0] = np.zeros(w2v_dim)

    label = np.empty((revs_size,), dtype="uint8")
    if(model_type == "CNN"):
        data = np.empty((revs_size, 1, max_len + 2*pad, w2v_dim), dtype = "float32")
        for i in range(revs_size):
            rev = revs[i]
            s2v = get_sen_w2v(rev["text"], word_idx_map, w2v,max_len, filter_h)
            data[i,0,:,:] = s2v
            label[i] = int(rev["label"])
        label_class = len(set(label))

    elif (model_type == "RNN"):
        data = np.empty((revs_size, (max_len + 2*pad) ,  w2v_dim), dtype = "float32")
        #data = np.empty((revs_size,1, (max_len + 2*pad) *  w2v_dim), dtype = "float32")
        for i in range(revs_size):
            rev = revs[i]
            s2v = get_sen_w2v(rev["text"], word_idx_map, w2v,max_len, filter_h)
            data[i,:,:] = s2v
            #data[i,0,:] = [x for X in s2v for x in X]
            label[i] = int(rev["label"])
        label_class = len(set(label))


    return input_shape, w2v_dim, label_class, data, label

#sen: sentence
#w2v: word to vector
#s2v: sententce to vector
def get_sen_w2v(sent, word_idx_map, w2v, max_len, filter_h=5):
    words = sent.split()
    s2v = []
    pad =  filter_h - 1
    for i in xrange(pad):
        s2v.append(w2v[0])
    for word in words:
        if word in word_idx_map:
            s2v.append(w2v[ word_idx_map[word] ])
    while len(s2v) <  max_len + 2*pad:
        s2v.append(w2v[0])

    return s2v


#count precision and recall
#wait to complete
def pre_recall_count(test_label, predict_label, label_class):
    assert len(test_label) == len(predict_label), "test_label == predict_label"
    pre = np.zeros([label_class,2])
    recall = np.zeros([label_class,2])

    for i in range(0,len(test_label)):
        if(test_label[i] == predict_label[i]):
            pre[predict_label[i]][0] +=1
            pre[predict_label[i]][1] +=1
            recall[predict_label[i]][0] +=1
            recall[predict_label[i]][1] +=1
        elif(test_label[i] != predict_label[i]):
            pre[predict_label[i]][0]+=1
            recall[test_label[i]][0]+=1

    return pre, recall

def write_pre_loss(log, path, train_num, nb_epoch, seed):
    argv0_list = sys.argv[0].split("\\");
    script_name = argv0_list[len(argv0_list) - 1]; # get script file name self
    script_name = script_name[0:-3];
    path = path + "-" + script_name
    output_file = open(path, 'w+')
    output_file.write("log begin in %s\t%s\n"%(time.asctime( time.localtime(time.time()) ),seed ));
    name_list = []

    output_file.write("%d\t"%(train_num))
    for x in log.history:
        name_list.append(x)
        output_file.write("%s\t"%(x))
    output_file.write('\n')


    for index in range(0, nb_epoch):
        for name in name_list:
            output_file.write("%.4f\t"%(log.history[ name ][ index ]))
        output_file.write("%d\n"%(index+1))

    output_file.write("log over in %s\t%s\n"%(time.asctime( time.localtime(time.time()) ),seed ));
    output_file.close()
    return





if __name__=="__main__":    
    if (sys.argv[1] == "test"):
        #test function:
        exit()

    #config
    w2v_file = sys.argv[1]     
    data_folder = ["../data/rt-polarity.pos","../data/rt-polarity.neg"]    
    w2v_dim = 300

    #Begin
    print "loading data...",        
    revs, vocab = build_data_cv(data_folder, clean_string=True)
    #find the longest sentences as the input feature, shorter than it
    #will  be pad by zero
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    not_in_vocab =add_unknown_words(w2v, vocab)
    print "num words not in word2vec: " + str(not_in_vocab)
    print "rate: %.4f" %(not_in_vocab/len(vocab))

    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    """
    revs: list.     Store the all informaiton of sentences: label, text, num_words
    vocab: dict: word, nums.    the dict to store all appeared word in train_test data
    word_idx_map: dict: word, word_id.
    W: dict: word_id, word_vector. part of vector from Google 300 vector
    W2: dict: word_id, word_vector. All random vector, in range(-0.25, 0.25)

    """
    cPickle.dump([max_l, w2v_dim, revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    print "dataset created!"
    
