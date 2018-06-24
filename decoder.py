import tensorflow as tf
from gated_linear_units import CausalGatedLinearUnit, GatedLinearUnit


class TemporalDecoderBlock(tf.layers.Layer):


    def __init__(self, kernel_size,
                       filter_size,
                       embedding_size,
                       inference=False,
                       trainable=True,
                       name=None,
                       dtype=None,
                       activity_regularizer=None,
                       padding="same",
                       kernel_initializer=None,
                       **kwargs):
            super(TemporalDecoderBlock, self).__init__(trainable=trainable,
                dtype=dtype,
                activity_regularizer=activity_regularizer,
                name=name,
                **kwargs)


            self._embedding_size = embedding_size
            self._filter_size = filter_size

            if not inference:
                print("LOLOLOL")
                # actual 1D gated convolution
                self._causal_gated_layer = CausalGatedLinearUnit(filters=filter_size, kernel_size=kernel_size, padding=padding)
            else:
                self._causal_gated_layer = GatedLinearUnit(filters=filter_size, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer)
            # project from conv hidden state size ("d") to embedding size (used for attention)
            self._attention_layer = tf.layers.Conv1D(kernel_size=1, filters=embedding_size, activation=None, use_bias=True, name="attention")


    def call(self, decoder_inputs, proj_encoder_out, tgt_out):
        """At each step we take in the inputs to this decoder layer, the
            final encoder outputs and translation inputs
        """
        proj_encoder_out, proj_encoder_out_with_embeddings = proj_encoder_out
        #proj_encoder_out = tf.Print(proj_encoder_out, [proj_encoder_out])
        # h_i with residual connection
        gated_output = self._causal_gated_layer(decoder_inputs)# + decoder_inputs

        proj_gated_output = tf.layers.conv1d(gated_output, filters=self._embedding_size, kernel_size=1, activation=None, use_bias=False) if self._filter_size != self._embedding_size else gated_output

        # d_i = W * h_i + b + g_i
        attention_d = self._attention_layer(proj_gated_output) + tgt_out

        proj_encoder_out_transpose = tf.transpose(proj_encoder_out, perm=[0, 2, 1])
        # dot products, transpose z so that time is the column axis
        attention_mat = tf.nn.softmax(tf.matmul(attention_d, proj_encoder_out_transpose), axis=2)

        # compute the conditional input
        conditional_input = tf.matmul(attention_mat, proj_encoder_out_with_embeddings)

        proj_conditional_input = tf.layers.conv1d(conditional_input, filters=self._filter_size, kernel_size=1, activation=None, use_bias=False) if self._filter_size != self._embedding_size else conditional_input

        return gated_output + proj_conditional_input



class TemporalDecoder(tf.layers.Layer):

    def __init__(self, num_layers,
                 kernel_size,
                 filter_size,
                 embedding_size,
                 trainable=True,
                 name=None,
                 dtype=None,
                 activity_regularizer=None,
                 **kwargs):
        super(TemporalDecoder, self).__init__(trainable=True,
            dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name,
            **kwargs)

        self._embedding_size = embedding_size
        self._filter_size = filter_size
        self._layers = []
        for i in range(num_layers):
            self._layers.extend([TemporalDecoderBlock(kernel_size=kernel_size, filter_size=filter_size, embedding_size=embedding_size)])


    def call(self, tgt_in, proj_encoder_out, tgt_out):
        proj_tgt_in = tf.layers.conv1d(tgt_in, self._filter_size, kernel_size=1, activation=None, use_bias=False) if self._filter_size != self._embedding_size else tgt_in
        output = proj_tgt_in
        for layer in self._layers:
            output = layer(output, proj_encoder_out, tgt_out) + output

        return output
