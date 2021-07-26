import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_embeddings(file_path) :
    '''
    parameters : file_path - path of file where embeddings are stored (eg: '<path>/glove.6B/glove.6B.100d.txt')
    load the words and their respective embeddings from the GloVe file and set up a dictionary mapping 
    of words and their corresponding embeddings (embeddings will be stored in a numpy array of shape (d, ))
    returns : embedding_dict - dictionary mapping of {word:embedding}
    ''' 
    
    embedding_dict = {}
    file = open(file_path)
    for line in file :
        data = line.split(" ")
        word = data[0]
        embedding = np.asarray(data[1:], dtype='float32')
        embedding_dict[word] = embedding
    file.close()
    return embedding_dict



def set_embedding_matrix(embedding_dict, vocabulary, embedding_dimension) :
    '''
    parameters : embedding dict - dictionary mapping of {word:embedding} 
                 vocabulary - list of words in the training dataset
                 embedding_dimension - dimension of word embeddings used in the model
    initialises the embedding matrix with the ith row corresponding to the embedding of the ith word in the vocabulary
    dimension of the embedding depends on the 
    returns : embedding_matrix of shape (n, d) where n is the number of words in the vocabulary and d is the 
              dimension of the embeddings
    '''
    
    embedding_matrix = np.random.normal(0, 0.1, (len(vocabulary) + 1, embedding_dimension))
    for i, word in enumerate(vocabulary) :
        if word in embedding_dict.keys() :
            word_embedding = embedding_dict[word]
            embedding_matrix[i] = word_embedding
    return embedding_matrix

    



def get_rand_embedding_layer(vocabulary_size, embedding_dimension) :
    '''
    parameters : vocabulary_size - size of vocabulary used
                 embedding_dimension - integer which indicated the dimension of the word embeddings
                 max_length - maximum length of the input to the model(eg : maximum length of an input sentence)
    creates the embedding layer with trainable set to true so that weights cannot be changed during training.
    Weights of the embedding layer follow normal distribution with mean=0, stddev=0.1
    returns : embedding_layer 
    '''
    
    embedding_matrix = np.random.normal(0, 0.1, (vocabulary_size, embedding_dimension))
    embedding_layer = layers.Embedding(input_dim=vocabulary_size,
                                      output_dim=embedding_dimension,
                                      weights=[embedding_matrix],
                                      trainable=True,
                                      name='embedding_rand')
    return embedding_layer


def get_static_embedding_layer(embedding_matrix) :
    '''
    parameters : embedding_matrix - numpy array of shape (n, d) used to set the weights of the embedding layer
                 max_length - maximum length of the input to the model(eg : maximum length of an input sentence)
    creates the embedding layer and sets its weights with trainable set to false 
    so that weights cannot be changed during training
    returns : embedding_layer 
    '''
    
    embedding_layer = layers.Embedding(input_dim=embedding_matrix.shape[0],
                                      output_dim=embedding_matrix.shape[1],
                                      weights=[embedding_matrix],
                                      trainable=False,
                                      name='embedding_static')
    return embedding_layer


def get_dynamic_embedding_layer(embedding_matrix) :
    '''
    parameters : embedding_matrix - numpy array of shape (n, d) used to set the weights of the embedding layer
                 max_length - maximum length of the input to the model(eg : maximum length of an input sentence)
    creates the embedding layer and sets its weights with trainable set to true 
    so that weights can be changed or fine-tuned during training
    returns : embedding_layer 
    '''
    
    embedding_layer = layers.Embedding(input_dim=embedding_matrix.shape[0],
                                      output_dim=embedding_matrix.shape[1],
                                      weights=[embedding_matrix],
                                      trainable=True,
                                      name='embedding_dynamic')
    return embedding_layer



class PositionalEmbedding(layers.Layer) :
    def __init__(self) :
        super(PositionalEmbedding, self).__init__()
        
    def call(self, word_embeddings) :
        '''
        parameters : word_embeddings - tensor of shape (num_turns, seq_len, embed_dim)
        returns : embeddings_with_position - tensor of shape (num_turns, seq_len, embed_dim)
        '''
        positional_embeddings = np.zeros((word_embeddings.shape[1], word_embeddings.shape[2]))
        for i  in range(positional_embeddings.shape[0]) :
            if i % 2 == 0 :
                positional_embeddings[i] = np.array([np.sin(i/(1000 ** (2 * j / positional_embeddings.shape[1]))) for j in range(positional_embeddings.shape[1])])
            else :
                positional_embeddings[i] = np.array([np.cos(i/(1000 ** (2 * j / positional_embeddings.shape[1]))) for j in range(positional_embeddings.shape[1])])
        
        positional_embeddings = np.repeat(positional_embeddings[np.newaxis, :, :], word_embeddings.shape[0], axis=0)
        
        return positional_embeddings + word_embeddings