import tensorflow as tf

class TemporalEncoderBlock(tf.layers.Layer):
    """Temporal Convolution Encoder Block
    """

    def __init__(self, num_channels,
                 kernel_size=2,
                 dropout=0.2,
                 trainable=True,
                 name=None,
                 dtype=None,
                 activation=tf.nn.relu6,
                 activity_regularizer=None,
                 **kwargs):
        super(TemporalEncoderBlock, self).__init__(trainable=trainable,
            dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name,
            **kwargs)

        self.temporal_conv = tf.layers.Conv1D(filters=num_channels,
            kernel_size=kernel_size,
            activation=activation,
            padding="SAME")

        self.batch_norm = tf.layers.BatchNormalization()

        self.pooled = tf.layers.MaxPooling1D(pool_size=2, strides=1)

    def call(self, inputs, training=True):

        temporal_conv = self.temporal_conv(inputs)
        batch_norm = self.batch_norm(temporal_conv)
        pooled = self.pooled(batch_norm)
        return pooled

class TemporalDecoderBlock(tf.layers.Layer):

    def __init__(self, num_channels,
                 kernel_size=2,
                 trainable=True,
                 name=None,
                 dtype=None,
                 activation=None,
                 activity_regularizer=None,
                 **kwargs):
        super(TemporalDecoderBlock, self).__init__(trainable=trainable,
            dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name,
            **kwargs)

        self.temporal_conv = tf.layers.Conv1D(filters=num_channels,
            kernel_size=kernel_size,
            activation=activation,
            padding="SAME")

        self.unpooled = tf.keras.layers.UpSampling1D(size=2)

    def call(self, inputs, training=True):

        temporal_conv = self.temporal_conv(inputs)
        unpooled = self.unpooled(temporal_conv)
        return unpooled

class TemporalEncoder(tf.layers.Layer):

    """Temporal Encoder Network"""

    def __init__(self, num_blocks,
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

        self._blocks = []

        for i in range(num_blocks):
            self._blocks.extend([TemporalEncoderBlock(num_channels=(32 * (2 ^ (i + 1))))])

    def call(self, inputs, training=True):
        output = inputs
        for block in self._blocks:
            output = block(output)
        return output

class TemporalDecoder(tf.layers.Layer):

    """Temporal Decoder Network"""

    def __init__(self, num_encoder_blocks,
                  num_blocks,
                  trainable = True,
                  name = None,
                  dtype = None,
                  activity_regularizer = None,
                  **kwargs):
         super(TemporalDecoder, self).__init__(trainable = trainable,
                                         dtype = dtype,
                                         activity_regularizer = activity_regularizer,
                                         name = name,
                                         **kwargs)

         self._blocks = []
         last_filter_size = 32 * (num_encoder_blocks)
         for i in range(num_blocks):
             self._blocks.extend([TemporalDecoderBlock(num_channels=last_filter_size / (2 * (i + 1)))])

    def call(self, inputs, training=True):
        output = inputs
        for block in self._blocks:
            output = block(output)
        return output
