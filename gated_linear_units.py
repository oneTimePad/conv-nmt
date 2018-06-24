import tensorflow as tf

class GatedLinearUnit(tf.layers.Conv1D):

    def __init__(self, filters,
                       kernel_size,
                       strides = 1,
                       dilation_rate = 1,
                       activation = None,
                       use_bias = True,
                       kernel_initializer = None,
                       bias_initializer = tf.zeros_initializer(),
                       kernel_regularizer = None,
                       bias_regularizer = None,
                       activity_regularizer = None,
                       kernel_constraint = None,
                       bias_constraint = None,
                       trainable = True,
                       name = None,
                       padding = "same",
                       **kwargs):

        super(GatedLinearUnit, self).__init__(filters = 2 * filters,
                                           kernel_size = kernel_size,
                                           strides = strides,
                                           padding = padding,
                                           data_format = "channels_last",
                                           dilation_rate = dilation_rate,
                                           activation = activation,
                                           use_bias = use_bias,
                                           kernel_initializer = kernel_initializer,
                                           bias_initializer = bias_initializer,
                                           kernel_regularizer = kernel_regularizer,
                                           bias_regularizer = bias_regularizer,
                                           activity_regularizer = activity_regularizer,
                                           kernel_constraint = kernel_constraint,
                                           bias_constraint = bias_constraint,
                                           trainable = trainable,
                                           name = name,
                                           **kwargs)

    def gate_conv(self, inputs):
        """Represents the gating operation of a Gated Linear Unit
            Args:
                inputs: TF input tensors
            Returns:
                output tensors
        """

        # chan dimension is last
        half_channel_dim = int(int(inputs.get_shape()[2]) /2)

        A = inputs[:,:,:half_channel_dim]
        B = inputs[:,:,half_channel_dim: 2 * half_channel_dim]

        return A * tf.nn.sigmoid(B)

    def call(self, inputs, training=True):

        conv_outputs = super(GatedLinearUnit, self).call(inputs)

        return self.gate_conv(conv_outputs)
    def build(self, input_shape):
        super(GatedLinearUnit, self).build(input_shape)


class CausalGatedLinearUnit(GatedLinearUnit):
        def __init__(self, filters,
                           kernel_size,
                           strides = 1,
                           dilation_rate = 1,
                           activation = None,
                           use_bias = True,
                           kernel_initializer = None,
                           bias_initializer = tf.zeros_initializer(),
                           kernel_regularizer = None,
                           bias_regularizer = None,
                           activity_regularizer = None,
                           kernel_constraint = None,
                           bias_constraint = None,
                           trainable = True,
                           name = None,
                           padding = "same",
                           **kwargs):

            super(CausalGatedLinearUnit, self).__init__(filters,
                                                           kernel_size,
                                                           strides = strides,
                                                           dilation_rate = dilation_rate,
                                                           activation = activation,
                                                           use_bias = use_bias,
                                                           kernel_initializer = kernel_initializer,
                                                           bias_initializer = bias_initializer,
                                                           kernel_regularizer = kernel_regularizer,
                                                           bias_regularizer = bias_regularizer,
                                                           activity_regularizer = activity_regularizer,
                                                           kernel_constraint = kernel_constraint,
                                                           bias_constraint = bias_constraint,
                                                           trainable = trainable,
                                                           name = name,
                                                           padding = padding,
                                                           **kwargs)

        def call(self, inputs, training=True):
            print("hello")
            padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
            inputs = tf.pad(inputs, tf.constant([(0, 0), (1, 0), (0, 0)]) * padding)
            inputs = tf.Print(inputs, [inputs], "Input is:")
            gated_conv = super(CausalGatedLinearUnit, self).call(inputs)

            return gated_conv
