import numpy as np
import time
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from _common_modules.embedding_layer import *
from _common_modules.transformer_modules import *

tf.config.run_functions_eagerly(True)


#only if GPU is available
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#data
meetings = np.load('../data/obj/meetings.npz')['arr_0'] # (num_meetings, num_turns, seq_len)
summary = np.load('../data/obj/summary.npz')['arr_0'] # (num_meetings, summary_len)

turns = np.load('../data/obj/turns.npz')['arr_0'] # (num_meetings, num_turns)
role_vector = np.load('../data/obj/role_vector.npz')['arr_0'] # (num_roles, MAX_LENGTH_BIN = 3)

sentence_embeddings = np.load('../data/obj/sentence_embeddings.npz')['arr_0'] # (num_meetings, num_turns, sentence_embedding_dim)

with open('../data/obj/tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)
    
vocabulary_size = len(tokenizer.word_index) + 1
word_embedding_dimension = 100
sentence_embedding_dimension = 100


class MTNet(tf.keras.Model) :
    def __init__(self, 
                 num_blocks, 
                 word_embedding_dimension,
                 sentence_embedding_dimension,
                 num_heads_word, 
                 num_heads_dec, 
                 vocabulary_size,
                 embedding_matrix,
                 role_vector_size, 
                 init_role_vector, 
                 num_clusters=50,
                 mode='static') :
        
        super(MTNet, self).__init__()
        
        self.init_role_vector = init_role_vector
        self.num_clusters = num_clusters
        
        self.embedding_layer = None
        if mode == 'static' :
            self.embedding_layer = get_static_embedding_layer(embedding_matrix)
        elif mode == 'dynamic' :
            self.embedding_layer = get_dynamic_embedding_layer(embedding_matrix)
        elif mode == 'rand' :
            self.embedding_layer = get_rand_embedding_layer(vocabulary_size, word_embedding_dimension) 
            
        self.positional_embedding_layer = PositionalEmbedding()
                                                            
        
        self.encoder = Encoder(num_blocks, 
                                          sentence_embedding_dimension + role_vector_size, 
                                          num_heads_word)
        
        self.role_vector = layers.Dense(role_vector_size)
        
        self.decoder = Decoder(num_blocks, 
                               word_embedding_dimension, 
                               num_heads_dec)
        
        self.fully_connected_layer = layers.Dense(vocabulary_size)
        
    def call(self, sentence_embedding, input_tensor, target_tensor, turn_seq) :

        sentence_embedding_input = self.positional_embedding_layer(sentence_embedding) # (1, num_turns, sent_embed_dim)
        
        input_role_vector = self.role_vector(self.init_role_vector) # (num_roles, role_vector_size)

        # (1, num_turns, sent_embed_dim + role_vector_size)
        x1_concat = concat_role_vector(sentence_embedding_input, input_role_vector, turn_seq) 
        
        x1 = self.encoder(x1_concat) # (1, num_turns, sent_embed_dim + role_vector_size)

        #for a single meeting or batch
        closest_points_index = get_clusters(self.num_clusters, x1[0]) # (num_turns)
        
        clustered_turns = np.zeros((self.num_clusters, input_tensor.shape[-1])) # (num_clusters, seq_len)

        for index, cluster_point in enumerate(closest_points_index) :
            clustered_turns[index] = input_tensor[0][cluster_point]
        
        x2 = self.embedding_layer(clustered_turns) # (num_clusters, seq_len, word_embed_dim)
        x2 = tf.reshape(x2, [1, -1, x2.shape[-1]]) # (1, num_clusters * seq_len, word_embed_dim)
        
        target_x = self.embedding_layer(target_tensor) # (batch_size=1, target_seq_len, word_embed_dim)
        target_x = self.positional_embedding_layer(target_x) # (batch_size=1, target_seq_len, word_embed_dim)
        
        x = self.decoder(target_x, x1, x2) # (batch_size=1, target_seq_len, embed_dim)
        x = self.fully_connected_layer(x)
        
        return x



trial_role_vector = tf.random.uniform((2, 2))
turn_seq = np.array([0,1,1,0,1,0,1,0,1,0])
temp_embedding_matrix = np.random.normal(0, 0.1, (vocabulary_size, word_embedding_dimension))

sample_mtnet = MTNet(num_blocks=2, 
                     word_embedding_dimension=100, 
                     sentence_embedding_dimension=100,
                     num_heads_word=11, 
                     num_heads_dec=10, 
                     vocabulary_size=vocabulary_size,
                     role_vector_size=32, 
                     init_role_vector=trial_role_vector, 
                     embedding_matrix=temp_embedding_matrix,
                     num_clusters=5,
                     mode='static')

sentence_embedding = tf.random.uniform((1, 10, 100), dtype=tf.float64, minval=0, maxval=200)
temp_input = tf.random.uniform((1, 10, 5), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((1, 3), dtype=tf.int64, minval=0, maxval=200)

fn_out = sample_mtnet(sentence_embedding, temp_input, temp_target, turn_seq) # (batch_size, tar_seq_len, vocab_size)


dataset = tf.data.Dataset.from_tensor_slices((sentence_embeddings, meetings, summary))
dataset = dataset.batch(1)


# load Glove embeddings(100 dimensional) and convert it into a dictionary with mapping {word:embedding}

file_path = '../../GloVe/glove.6B/glove.6B.100d.txt'
embedding_dict = load_embeddings(file_path)


vocabulary = tokenizer.word_index.keys()
embedding_matrix = set_embedding_matrix(embedding_dict, vocabulary, 100)


optimizer = tf.keras.optimizers.Adam(0.1, beta_1=0.9, beta_2=0.98,epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')



def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


mtnet = MTNet(num_blocks=2, 
             word_embedding_dimension=100,
             sentence_embedding_dimension =100,
             num_heads_word=11, 
             num_heads_dec=10, 
             vocabulary_size=vocabulary_size,  
             embedding_matrix=embedding_matrix,
             role_vector_size=32, 
             init_role_vector=role_vector,
             num_clusters=5,
             mode='static')


EPOCHS = 20

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(1, 70, 100), dtype=tf.float32),
    tf.TensorSpec(shape=(1, 70, 100), dtype=tf.int64),
    tf.TensorSpec(shape=(1, 100), dtype=tf.int64),
    tf.TensorSpec(shape=(70), dtype=tf.int64),
]

@tf.autograph.experimental.do_not_convert
@tf.function(input_signature=train_step_signature)
def train_step(sentence_embeddings, input_tensor, target_tensor, turn_seq):
        
    target_inp = target_tensor[:, :-1]
    target_real = target_tensor[:, 1:]


    with tf.GradientTape() as tape:
        predictions = mtnet(sentence_embeddings, input_tensor, target_inp, turn_seq)
        loss = loss_function(target_real, predictions)

    gradients = tape.gradient(loss, mtnet.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mtnet.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(target_real, predictions))


for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (sentence_embeddings1, meetings1, summary1)) in enumerate(dataset):

        train_step(sentence_embeddings1, meetings1, summary1, turns[batch])

        if batch % 5 == 0:
            print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')


    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

