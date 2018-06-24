import tensorflow as tf
from gated_linear_units import GatedLinearUnit


class TemporalEncoder(tf.layers.Layer):

    def __init__(self, num_layers,
                 kernel_size,
                 filter_size,
                 embedding_size,
                 trainable = True,
                 name = None,
                 dtype = None,
                 activity_regularizer = None,
                 **kwargs):
        super(TemporalEncoder, self).__init__(trainable = trainable,
                                            dtype = dtype,
                                            activity_regularizer = activity_regularizer,
                                            name = name,
                                            **kwargs)
        self.kernel_size = kernel_size
        self._layers = []
        self._filter_size = filter_size
        self._embedding_size = embedding_size

        for i in range(num_layers):
            self._layers.extend([GatedLinearUnit(filters=filter_size, kernel_size=kernel_size)])

    def call(self, inputs):

        # project embeddings into conv output space
        output = tf.layers.conv1d(inputs, filters=self._filter_size, kernel_size=1, activation=None, use_bias=False) if self._filter_size != self._embedding_size else inputs

        for layer in self._layers:#layer, filter_size in self._layers:#zip(self._layers, self._filter_sizes):
            # add residual connections
            output = layer(output) + output

        proj_outputs = tf.layers.conv1d(output, kernel_size=1, filters=self._embedding_size, use_bias=False, activation=None) if self._filter_size != self._embedding_size else output

        # encoder projected output and encoder output + positional embeddings
        return proj_outputs,  proj_outputs + inputs
