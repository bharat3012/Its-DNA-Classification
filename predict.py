from keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda
from keras.engine import Input, Model, InputSpec
from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.models import model_from_json
from keras import backend as K
from sklearn import decomposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
import argparse

CHUNK_SIZE = 150


"""providing the index to each letter in an alphabet example in _alphabet A is at 0 index T at 1 so 
when we call letter_to_index('T') it will return index of T =1"""
def letter_to_index(letter):
    _alphabet = 'ATGC'
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter), None)




"""input_file is nothing but by default if test.fasta which contains a sequence of ATGC of size (1,603) and having squence as heade"""
""" then converting fasta file into csv file"""
def make_chunks_from_fasta(input_file):
    print ('Loading data...')
    # create fasta fragments with size of 150-----
    # Creating a file named input_fragments
    with open('input_fragments.csv',"w") as f:
        f.write(str('sequence\n'))
        #SeqIO.parse() is for reading a input_file -"text.fasta"
        for seq_record in SeqIO.parse(input_file, "fasta"):
            #seq_record.seq will give all the letters GGCT.....CGT whose length is 603
            # so create fasta fragments of size 150 in a new line and store it into input_fragments.csv
            for i in range(0, len(seq_record.seq) - int(CHUNK_SIZE) - 1, CHUNK_SIZE) :
               #i be like - 0, 149, 299.. 
               f.write(str(seq_record.seq[i:i + int(CHUNK_SIZE)]) + "\n")
    with open('input_names.csv',"w") as f:
        f.write(str('name\n'))
        for seq_record in SeqIO.parse(input_file, "fasta"):
            for i in range(0, len(seq_record.seq) - int(CHUNK_SIZE) - 1, CHUNK_SIZE) :
                #write id
                f.write(str(seq_record.id) + "\n")
#get ids and seperate each with a comma
def get_ids():
    dat = pd.read_csv('input_names.csv', sep=",")
    return dat


#loading of a test file----that means 
    """ input_fragments.csv """
def load_test(input_file):
    print ('Loading data...')
    
    #read a test csv file 
    df = pd.read_csv(input_file)
    #to the header sequence 
    df['sequence'] = df['sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    
    #set in random order of no.
    df = df.reindex(np.random.permutation(df.index))
    
    #Values to each word of a sequence  
    sample = df['sequence'].values[:len(df)]
    
    """ pad_sequences insures the list must contain equal size of sequences - like pad_sequences([1,2,3],[4,7,8,9],[7,8])
    returns - array([0,1,2,3]
                    [4,7,8,9]
                    [0,0,7,8])"""
    #padding to every seuence
    return pad_sequences(sample)


"""presently no use"""
def visualize_model(model, include_gradients=False):
    
    #retreives a layer based on the name 
    recurrent_layer = model.get_layer('recurrent_layer')
    output_layer = model.get_layer('output_layer')
    
    #initialize list
    inputs = []
    
    # extend is similar to append which makes complete output as one list not a list of list
    inputs.extend(model.inputs)
    
    #
    outputs = []
    outputs.extend(model.outputs)
    
    outputs.append(recurrent_layer.output)
    outputs.append(recurrent_layer.trainable_weights[1])  # -- weights of the forget gates (assuming LSTM)
    if include_gradients:
        loss = K.mean(model.output)  # [batch_size, 1] -> scalar
        grads = K.gradients(loss, recurrent_layer.output)
        grads_norm = grads / (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        outputs.append(grads_norm)
    all_function = K.function(inputs, outputs)
    output_function = K.function([output_layer.input], model.outputs)
    return all_function, output_function


"""presently No use use"""
def get_compare_embeddings(original_embeddings, tuned_embeddings, vocab, dimreduce_type="pca", random_state=0):
    """ Compare embeddings drift. """
    # Dimensional reduction they are using PCA where t-SNE could be a better alternative
    if dimreduce_type == "pca":
        # import PCA from sklearn
        from sklearn.decomposition import PCA
        dimreducer = PCA(n_components=2, random_state=random_state)
    elif dimreduce_type == "tsne":
        from sklearn.manifold import TSNE
        dimreducer = TSNE(n_components=2, random_state=random_state)
    else:
        raise Exception("Wrong dimreduce_type.")
        
       
        
    reduced_original = dimreducer.fit_transform(original_embeddings)
    reduced_tuned = dimreducer.fit_transform(tuned_embeddings)
    def compare_embeddings(word):
        if word not in vocab:
            return None
        word_id = vocab[word]
        original_x, original_y = reduced_original[word_id, :]
        tuned_x, tuned_y = reduced_tuned[word_id, :]
        return original_x, original_y, tuned_x, tuned_y
    return compare_embeddings


#args just to parse the variables

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', action='store', dest='input', help='Path to input FASTA file (.fasta)', default='test.fasta')
    parser.add_argument('-m', '--model', action='store', dest='model', help='Path to model (.json)', default='model.json')
    parser.add_argument('-w', '--weights', action='store', dest='weights', help='Path to model weights (.h5)', default='model.h5')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    args = parser.parse_args()
 
    print ('Loading model...')
    # We are first saving rnn model into a json file in train.py and here we are opening that saved file
    json_file = open(args.model, 'r')
    #reading that model.json file
    loaded_model_json = json_file.read()
    json_file.close()
    
    #keras function to adopt json model into keras format
    model = model_from_json(loaded_model_json)
    # load weights into new model
    
    print("Load model from disk")
    #load weights(parameters) from model.h5 file which was saved using train.py
    model.load_weights(args.weights)
    
    
    #input_file here is test.data - makes fasta file into csv with the sequence divided into chunks of size 150 (603/150)
    print("Make chunks")
    make_chunks_from_fasta(args.input)
    
    
    # loading the created csv file and apply indexing and padding on it
    print('Predict samples...')
    X = load_test('input_fragments.csv')
    
    #using keras function predict after loading csv file we get predicted y(output)
    y = model.predict(X, verbose=0)
    ids = get_ids()#seperate by commas
    probabilities = y[:,0]
    df = pd.concat([ids, pd.DataFrame(probabilities)], axis=1)# storing the output
    df.to_csv('full_report.txt', index=False)
    df = df.set_index(['name']).stack().groupby(level=0).agg('mean')
    
    #masking probabilities using threshold 0.5 and storing it in report_string.txt
    df_masked = df.mask(df > .5, 'plasmid')
    df_masked = df_masked.mask(df_masked <= .5, 'chromosome')
    df_masked.to_csv('report_string.txt', index=True)

    print(df_masked)