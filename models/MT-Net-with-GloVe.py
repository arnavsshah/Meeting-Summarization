import numpy as np
import time
import pickle

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

with open('../data/obj/tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)
    
vocabulary_size = len(tokenizer.word_index) + 1
embedding_dimension = 100



class MTNet(tf.keras.Model) :
    def __init__(self, 
                 num_blocks, 
                 embedding_dimension, 
                 num_heads_word, 
                 num_heads_turn, 
                 num_heads_dec, 
                 vocabulary_size,
                 embedding_matrix,
                 role_vector_size, 
                 init_role_vector, 
                 mode='static') :
        
        super(MTNet, self).__init__()
        
        self.init_role_vector = init_role_vector
                
        self.embedding_layer = None
        if mode == 'static' :
            self.embedding_layer = get_static_embedding_layer(embedding_matrix)
        elif mode == 'dynamic' :
            self.embedding_layer = get_dynamic_embedding_layer(embedding_matrix)
        elif mode == 'rand' :
            self.embedding_layer = get_rand_embedding_layer(vocabulary_size, embedding_dimension) 
            
        self.positional_embedding_layer = PositionalEmbedding()
                                                            
        
        self.word_level_encoder = Encoder(num_blocks, 
                                          embedding_dimension, 
                                          num_heads_word)
        
        self.turn_level_encoder = Encoder(num_blocks, 
                                          embedding_dimension + role_vector_size, 
                                          num_heads_turn)
        
        self.role_vector = layers.Dense(role_vector_size)
        
        self.decoder = Decoder(num_blocks, 
                               embedding_dimension, 
                               num_heads_dec)
        
        self.fully_connected_layer = layers.Dense(vocabulary_size)
        
    def call(self, input_tensor, target_tensor, turn_seq) :
        
        input_tensor = tf.squeeze(input_tensor, [0]) # (num_turns, seq_len)
        
        embedding_input = self.embedding_layer(input_tensor) # (num_turns, seq_len, embed_dim)
        embedding_input = self.positional_embedding_layer(embedding_input) # (num_turns, seq_len, embed_dim)
        
        x1 = self.word_level_encoder(embedding_input) # (num_turns, seq_len, embed_dim)

        input_role_vector = self.role_vector(self.init_role_vector) # (num_roles, role_vector_size)

        x1_concat = concat_role_vector(x1, input_role_vector, turn_seq) # (num_turns, 1, embed_dim + role_vector_size)

        x2 = self.turn_level_encoder(x1_concat)
        
        x1 = tf.reshape(x1, [1, -1, x1.shape[-1]]) # (1, num_turns * seq_len, embed_dim)
        x2 = tf.reshape(x2, [1, -1, x2.shape[-1]]) # (1, num_turns * seq_len, embed_dim)
        
        
        target_x = self.embedding_layer(target_tensor) # (num_turns, seq_len, embed_dim)
        target_x = self.positional_embedding_layer(target_x) # (num_turns, seq_len, embed_dim)
        
        x = self.decoder(target_x, x1, x2) #(batch_size=1, target_seq_len, embed_dim)
        x = self.fully_connected_layer(x)
        
        return x



trial_role_vector = tf.random.uniform((2, 2))
turn_seq = np.array([0,1,1,0,1,0,1,0,1,0])
temp_embedding_matrix = np.random.normal(0, 0.1, (vocabulary_size, embedding_dimension))

sample_mtnet = MTNet(num_blocks=2, 
                     embedding_dimension=100, 
                     num_heads_word=10, 
                     num_heads_turn=11,
                     num_heads_dec=10, 
                     vocabulary_size=vocabulary_size,
                     role_vector_size=32, 
                     init_role_vector=trial_role_vector, 
                     embedding_matrix=temp_embedding_matrix,
                     mode='static')

temp_input = tf.random.uniform((1, 10, 5), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((1, 3), dtype=tf.int64, minval=0, maxval=200)

fn_out = sample_mtnet(temp_input, temp_target, turn_seq) # (batch_size, tar_seq_len, vocab_size)



dataset = tf.data.Dataset.from_tensor_slices((meetings, summary))
dataset = dataset.batch(1)




# load Glove embeddings(100 dimensional) and convert it into a dictionary with mapping {word:embedding}

file_path = '../GloVe/glove.6B/glove.6B.100d.txt'
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
             embedding_dimension=100, 
             num_heads_word=5, 
             num_heads_turn=5,
             num_heads_dec=5, 
             vocabulary_size=vocabulary_size,  
             embedding_matrix=embedding_matrix,
             role_vector_size=10, 
             init_role_vector=role_vector,
             mode='static')



EPOCHS = 20

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(1, 70, 100), dtype=tf.int64),
    tf.TensorSpec(shape=(1, 100), dtype=tf.int64),
    tf.TensorSpec(shape=(70), dtype=tf.int64),
]

@tf.autograph.experimental.do_not_convert
@tf.function(input_signature=train_step_signature)
def train_step(input_tensor, target_tensor, turn_seq):

    target_inp = target_tensor[:, :-1]
    target_real = target_tensor[:, 1:]


    with tf.GradientTape() as tape:
        predictions = mtnet(input_tensor, target_inp, turn_seq)
        loss = loss_function(target_real, predictions)

    gradients = tape.gradient(loss, mtnet.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mtnet.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(target_real, predictions))


for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (meetings1, summary1)) in enumerate(dataset):

        train_step(meetings1, summary1, turns[batch])

        if batch % 5 == 0:
            print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')





