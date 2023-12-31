from keras.layers import RepeatVector, Layer


class Bottleneck(Layer):
    rnn_layer: Layer
    repeat_layer: RepeatVector

    def __init__(self, rnn: Layer, time_steps: int, **kwargs):
        """
        Creates a bottleneck layer.

        :param rnn: The RNN layer to use.
        :type rnn: Layer
        :param time_steps: The number of time steps.
        :type time_steps: int
        :param kwargs: The keyword arguments passed to the Layer constructor.
        :type kwargs: dict
        """
        self.rnn_layer = rnn
        self.repeat_layer = RepeatVector(time_steps)
        super(Bottleneck, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        return self.repeat_layer(self.rnn_layer(inputs))
    
    def compute_mask(self, inputs, mask=None):
        return mask
