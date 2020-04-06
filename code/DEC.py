import keras.backend as K
from keras.engine.topology import Layer,InputSpec
from keras.layers import Dense,Input,Dropout, BatchNormalization
from keras.models import Model
from sklearn.cluster import KMeans
import numpy as np


def autoencoder(dims, act='relu',init='glorot_uniform', dropout=False):
    '''
    This function return a encoder-decoder model and the encoder part of that model(SAE)
    Arguments:
            dims: a list contain the dimision of different layers. 
                  In the paper, dims is [x.shape[1],500,500,2000,10]
            activation: activation function in different layers.
                  In the paper, relu is recommended
            init: initialize method.
                  In the paper, just use a gussian dist with 0 mean and 0.01 std
            dropout: whether to use dropout
                  In the paper, dropout percentage is 20%
    return:
            two model: First is the encoder-decoder model. Second is the encoder model
    '''
    n_stacks = len(dims) - 1

    input_data = Input(shape=(dims[0],), name="input")
    x = input_data

    # internal layers of encoder
    for i in range(n_stacks - 1):
        if dropout:
            x = Dropout(rate=dropout)(x)
        x = Dense(
            dims[i + 1], activation=act, kernel_initializer=init, name="encoder_%d" % i
        )(x)
        x=BatchNormalization()(x)

    # latent hidden layer
    encoded = Dense(
        dims[-1], kernel_initializer=init, name="encoder_%d" % (n_stacks - 1)
    )(x)

    x = encoded
    # internal layers of decoder
    for i in range(n_stacks - 1, 0, -1):
        if dropout:
            x = Dropout(rate=dropout)(x)
        x=BatchNormalization()(x)
        x = Dense(
            dims[i], activation=act, kernel_initializer=init, name="decoder_%d" % i
        )(x)

    # decoder output

    x = Dense(dims[0], kernel_initializer=init, name="decoder_0")(x)

    decoded = x

    autoencoder_model = Model(inputs=input_data, outputs=decoded, name="autoencoder")
    encoder_model = Model(inputs=input_data, outputs=encoded, name="encoder")

    return autoencoder_model, encoder_model

def target_distribution(q):
    '''
    q is the output of ClusteringLayer
    '''
    weight = q**2/q.sum(0)
    return (weight.T / weight.sum(1)).T

class ClusteringLayer(Layer):
    '''
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    '''

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(name='clusters', shape=(self.n_clusters, input_dim), initializer='glorot_uniform') 
        
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        ''' 
        student t-distribution, as used in t-SNE algorithm.
        It measures the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
       
        inputs: the variable containing data, shape=(n_samples, n_features)
        
        Return: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        '''
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure all of the values of each sample sum up to 1.
        
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


