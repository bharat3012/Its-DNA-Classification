import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda, Bidirectional
from keras.engine import Input, Model, InputSpec
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import model_from_json
import os
#import pydot
#import graphviz

EPCOHS = 100 #  an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
BATCH_SIZE = 500 # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.
INPUT_DIM = 4 # a vocabulary of 4 words in case of fnn sequence (ATCG)
OUTPUT_DIM = 50 # Embedding output
RNN_HIDDEN_DIM = 62
DROPOUT_RATIO = 0.2 # proportion of neurones not used for training
MAXLEN = 150 # cuts text after number of these characters in pad_sequences
checkpoint_dir ='checkpoints'
os.path.exists(checkpoint_dir)

input_file = 'train_cami.csv'

"""convert each letters into indexes------This is most important"""
def letter_to_index(letter):
    _alphabet = 'ATGC'
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter), None)


"""load train_cami_r.csv file --- read it using pandas --- it has a header string 'sequence' using this apply indexing to them using"""
def load_data(test_split = 0.1, maxlen = MAXLEN):
    print ('Loading data...')
    df = pd.read_csv(input_file)
    
    #indexing to csv file on each letter
    df['sequence'] = df['sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])\
    
    """random shuffling and of above indexes obtained and reindexing it"""
    df = df.reindex(np.random.permutation(df.index))
    
    """ use training daata - 90% percent of original data"""
    train_size = int(len(df) * (1 - test_split))
    
    
    X_train = df['sequence'].values[:train_size]
    y_train = np.array(df['target'].values[:train_size])
    
    """test set"""
    X_test = np.array(df['sequence'].values[train_size:])
    y_test = np.array(df['target'].values[train_size:])
    print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))
    return pad_sequences(X_train, maxlen=maxlen), y_train, pad_sequences(X_test, maxlen=maxlen), y_test


""" Just pass 1st column length of X_train a input_length and then use Bi-RNN model"""
def create_lstm(input_length, rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO):
    model = Sequential()
    model.add(Embedding(input_dim = INPUT_DIM, output_dim = output_dim, input_length = input_length, name='embedding_layer'))
    model.add(Bidirectional(LSTM(rnn_hidden_dim, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(rnn_hidden_dim)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

def create_plots(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    plt.clf()

if __name__ == '__main__':
    # train
    """Create and Split Train and test Data """
    X_train, y_train, X_test, y_test = load_data()  
    """pass the length of X_train"""
    model = create_lstm(len(X_train[0])) 

    # save checkpoint
    filepath= checkpoint_dir + "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5" #filename where to save weights
    
    #save the model after every epoch using ModelCheckpoint- allows where to checkpoint the weights-- 
    #API allows you to specify which metric to monitor, such as loss or accuracy on the training or validation dataset.- here "val_acc"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')#using keras ModelCheckpoints
    callbacks_list = [checkpoint]# make a list of it
    
    
    # class_weight ='balanced' then class_weights -- n_samples / (n_classes * np.bincount(y)) and classes= np.unique(y_train), array-class_labels
    print ('Fitting model...')
    class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    print(class_weight)
    
    #train a network using keras and store in history
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, class_weight=class_weight,
        epochs=EPCOHS, callbacks=callbacks_list, validation_split = 0.1, verbose = 1)

    # serialize model to JSON---for further use in predicting using predict.py
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5---for further use in predicting using predict.py
    model.save_weights("model.h5")
    print("Saved model to disk")
    create_plots(history)
    plot_model(model, to_file='model.png')

    # validate model on unseen data
    score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print('Validation score:', score)
    print('Validation accuracy:', acc)
