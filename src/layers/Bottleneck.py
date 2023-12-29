import tensorflow as tf
from tensorflow.keras.layers import RepeatVector

class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, rnn, timesteps, **kwargs):
        self.time_steps = timesteps
        self.rnn_layer = rnn
        self.repeat_layer = RepeatVector(timesteps)
        super(Bottleneck, self).__init__(**kwargs)
    
    def call(self, inputs):
        return self.repeat_layer(self.rnn_layer(inputs))
    
    def compute_mask(self, inputs, mask=None):
        return mask