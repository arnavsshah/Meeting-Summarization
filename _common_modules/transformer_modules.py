import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ScaledDotProductAttention(layers.Layer) :
    def __init__(self, is_mask=False) :
        super(ScaledDotProductAttention, self).__init__()
        self.is_mask = is_mask
        
    def call(self, query, key, value) :
        '''
        parameters : query - tensor of shape (num_turns, num_heads, seq_len_q, dim) 
                     key - tensor of shape (num_turns, num_heads, seq_len_k, dim) 
                     value - tensor of shape (num_turns, num_heads, seq_len_v, dim) 
                     **seq_len_k == seq_len_v
        returns : attention - tensor of shape (num_turns, num_heads, seq_len, dim) 
        '''
        # (num_turns, num_heads, seq_len_q, seq_len_k)
        pre_attention = tf.linalg.matmul(query, key, transpose_b=True) / np.sqrt(key.shape[1])

        if self.is_mask is True :
            mask = np.zeros((pre_attention.shape[-2], pre_attention.shape[-1]))
            mask.fill(-1e10)            
            mask = np.triu(mask, k=1)
            pre_attention = tf.math.multiply(pre_attention, mask)
            
        attention_weights = tf.nn.softmax(pre_attention, axis=-1)
        
        # (num_turns, num_heads, seq_len_q, dim)
        attention = tf.linalg.matmul(attention_weights, value)
        
        return attention



class MultiHeadAttention(layers.Layer) :
    def __init__(self, embedding_dimension, num_heads, is_mask=False) :
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.embedding_dimension = embedding_dimension
        self.dim = self.embedding_dimension // self.num_heads
        self.is_mask = is_mask
        
        assert(self.dim * self.num_heads == self.embedding_dimension), "embedding_dimension should be divisible by num_heads."

        self.query_layer = layers.Dense(self.embedding_dimension)
        self.key_layer = layers.Dense(self.embedding_dimension)
        self.value_layer = layers.Dense(self.embedding_dimension)
        
        self.scaled_dot_product_attention = ScaledDotProductAttention(is_mask=self.is_mask)
        
        self.linear_layer = layers.Dense(self.embedding_dimension)
    
    
    def split_heads(self, input_tensor) :
        '''
        parameters : input_tensor - tensor of shape (num_turns, seq_len, embedding_dimension)
        returns : input_tensor - resize tensor of shape (num_turns, num_heads, seq_len, dim)
        '''
        input_tensor = tf.reshape(input_tensor, (input_tensor.shape[0], -1, self.num_heads, self.dim))
        return tf.transpose(input_tensor, [0,2,1,3])
        
        
    def call(self, query, key, value) :
        '''
        parameters : query - tensor of shape (num_turns, seq_len, embedding_dimension)
                     key - tensor of shape (num_turns, seq_len, embedding_dimension)
                     value - tensor of shape (num_turns, seq_len, embedding_dimension)
        returns : res - tensor of shape (num_turns, seq_len, embedding_dimension)
        '''

        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)
        
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        
        attention = self.scaled_dot_product_attention(query, key, value) # (num_turns, num_heads, seq_len, dim)
        
        attention = tf.transpose(attention, perm=[0, 2, 1, 3]) # (num_turns, seq_len, num_heads, dim)
        concat_attention = tf.reshape(attention, (attention.shape[0], -1, self.embedding_dimension)) # (num_turns, seq_len, embedding_dimension)
        
        res = self.linear_layer(concat_attention) # (num_turns, seq_len, embedding_dimension)
        return res



class AddandNorm(layers.Layer) :
    def __init__(self) :
        super(AddandNorm, self).__init__()
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, input_tensor, skip_connection) :
        '''
        parameters : input_tensor - tensor of shape (num_turns, seq_len, embedding_dimension)
                     skip_connection - tensor of shape (num_turns, seq_len, embedding_dimension)
        returns : res - normalized tensor of shape (num_turns, seq_len, embedding_dimension)
        '''
        res = input_tensor + skip_connection
        res = self.layer_norm(res)
        return res



class FeedForward(layers.Layer) :
    def __init__(self, hidden_dim, output_dim) :
        super(FeedForward, self).__init__()
        self.layer_1 = layers.Dense(hidden_dim, activation='relu')
        self.layer_2 = layers.Dense(output_dim)
        
    def call(self, input_tensor) :
        '''
        parameters : input_tensor - tensor of shape (num_turns, seq_len, embedding_dimension)
        returns : input_tensor - tensor of shape (num_turns, seq_len, embedding_dimension)
        '''
        res = self.layer_1(input_tensor)
        res = self.layer_2(res)
        return res



def concat_role_vector(x, role_vector, turn_seq) :
    '''
    parameters : x - tensor of shape (num_turns, seq_len, embed_dim)
                 role_vector - tensor of shape (num_roles, role_vector_size)
                 turn_seq - np array of shape (num_turns, ) representing the sequence of turns in a meeting
    returns : concat_vector - concatenated vector of '<BOS>' tag and role_vector for each turn
                              of shape (num_turns, 1, embed_dim + role_vector_size)
    '''

    turn = x[:, :1, :]

    unpacked_turn_seq = tf.unstack(turn_seq)
    
    role = tf.expand_dims(tf.convert_to_tensor([role_vector[j] for j in unpacked_turn_seq]), 1)
    turn_with_role = tf.concat([turn, role], axis=2) # (num_turns, 1, embed_dim + role_vector_size)
    
    return turn_with_role


def get_clusters(num_clusters, data) :
    '''
    parameters : num_clusters - number of clusters for K-means clustering
                 data - tensor of shape (num_turns, sent_embed_dim + role_vector_size)
    returns : closest_data - np array of shape (num_clusters, ) containing indices of top 'num_cluster' sentences
    '''
    
    num_clusters = num_clusters

    m_km = KMeans(n_clusters=num_clusters, random_state=42)
    m_km = m_km.fit(data.numpy())
    m_clusters = m_km.labels_.tolist()

    centers = np.array(m_km.cluster_centers_)

    closest_data = []
    for i in range(num_clusters):
        center_vec = centers[i]
        data_idx_within_i_cluster = [ idx for idx, clu_num in enumerate(m_clusters) if clu_num == i ]

        one_cluster_tf_matrix = np.zeros( (  len(data_idx_within_i_cluster) , centers.shape[1] ) )
        for row_num, data_idx in enumerate(data_idx_within_i_cluster):
            one_row = data[data_idx]
            one_cluster_tf_matrix[row_num] = one_row

        closest, _ = pairwise_distances_argmin_min(center_vec.reshape(1, -1), one_cluster_tf_matrix)
        closest_idx_in_one_cluster_tf_matrix = closest[0]
        closest_data_row_num = data_idx_within_i_cluster[closest_idx_in_one_cluster_tf_matrix]

        closest_data.append(closest_data_row_num)
    return np.sort(np.array(closest_data))      



class EncoderBlock(layers.Layer) :
    def __init__(self, embedding_dimension, num_heads) :
        super(EncoderBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dimension, num_heads)
        self.add_and_norm_1 = AddandNorm()
        self.feed_forward = FeedForward(200, embedding_dimension)
        self.add_and_norm_2 = AddandNorm()
        
    def call(self, input_tensor) :
        '''
        parameters : input_tensor - tensor of shape (num_turns, seq_len, embedding_dimension)
        returns : input_tensor - tensor of shape (num_turns, seq_len, embedding_dimension)
        '''
        res = self.multi_head_attention(input_tensor, input_tensor, input_tensor)
        res_skip = self.add_and_norm_1(res, input_tensor)
        res = self.feed_forward(res_skip)
        res = self.add_and_norm_2(res, res_skip)
        return res



class DecoderBlock(layers.Layer) :
    def __init__(self, embedding_dimension, num_heads) :
        super(DecoderBlock, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(embedding_dimension, num_heads, is_mask=True)
        self.add_and_norm_1 = AddandNorm()
        
        self.multi_head_attention_1 = MultiHeadAttention(embedding_dimension, num_heads)
        self.add_and_norm_2 = AddandNorm()
        
        self.multi_head_attention_2 = MultiHeadAttention(embedding_dimension, num_heads)
        self.add_and_norm_3 = AddandNorm()
        
        self.feed_forward = FeedForward(200, embedding_dimension)
        self.add_and_norm_4 = AddandNorm()
        
    def call(self, input_tensor, sentence_level_encoder_output, turn_level_encoder_output) :
        res = self.masked_multi_head_attention(input_tensor, input_tensor, input_tensor)
        res_skip = self.add_and_norm_1(res, input_tensor)
        
        res = self.multi_head_attention_1(res_skip, sentence_level_encoder_output, sentence_level_encoder_output)
        res_skip = self.add_and_norm_2(res, res_skip)
        
        res = self.multi_head_attention_2(res_skip, turn_level_encoder_output, turn_level_encoder_output)
        res_skip = self.add_and_norm_3(res, res_skip)
        
        res = self.feed_forward(res_skip)
        res = self.add_and_norm_4(res, res_skip)
        return res



class Encoder(layers.Layer) :
    def __init__(self, 
                 num_blocks, 
                 embedding_dimension, 
                 num_heads) :
        
        super(Encoder, self).__init__()
        self.num_blocks = num_blocks
        
        self.encoder_blocks = [EncoderBlock(embedding_dimension, num_heads) for _ in range(num_blocks)]
        
    def call(self, input_tensor) :
        '''
        parameters : input_tensor : tensor of shape (num_turns, seq_len, embed_dim) / (num_turns, 1, embed_dim + role_vector_size)
        returns : x - tensor of shape (num_turns, seq_len, embed_dim) / (num_turns, 1, embed_dim + role_vector_size)
        '''
        x = input_tensor
        
        for i in range(self.num_blocks) :
            x = self.encoder_blocks[i](x) # (num_turns, seq_len, embed_dim) 
            
        return x



class Decoder(layers.Layer) :
    def __init__(self, 
                 num_blocks, 
                 embedding_dimension, 
                 num_heads) :
        
        super(Decoder, self).__init__()
        self.num_blocks = num_blocks
        
        self.decoder_blocks = [DecoderBlock(embedding_dimension, num_heads) for _ in range(num_blocks)]
        
    def call(self, input_tensor, word_level_encoder_output, turn_level_encoder_output) :
        '''
        parameters : input_tensor : tensor of shape (batch_size=1, target_seq_len, embed_dim)
        returns : x - tensor of shape (batch_size=1, target_seq_len, embed_dim) 
        '''
        x = input_tensor
        for i in range(self.num_blocks) :
            x = self.decoder_blocks[i](x, word_level_encoder_output, turn_level_encoder_output)
            
        return x
