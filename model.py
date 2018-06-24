import tensorflow as tf
import abc
import collections
from hparams import RNNHyperParameters, TemporalCNNHyperParameters
from encoder import TemporalEncoder
from decoder import TemporalDecoder, TemporalDecoderBlock
import os

"""Represents a the collection of Neural Translation Models
   Classes each represent a type of model allowing for building,
   training, evaluating and inference of/on graphs.
"""

# Represents a handle to a NT model graph
ModelGraph = collections.namedtuple("ModelGraph", ["logits", "samples","loss", "update_step",
    "num_units_per_cell", "num_layers", "embeddings_size", "graph","max_iter"])


class NeuralTranslationModel(abc.ABC):
    """Abstract Class for constructing
    Encoder/Decoder Models
    """

    def __init__(self,
                 hparams,
                 src_vocab,
                 tgt_vocab=None):

        self.embeddings_size = hparams.embeddings_size

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.global_step = tf.Variable(0, name="global_step")

        self.time_major = hparams.time_major

        self.hparams = hparams

        self.saver = None

    def train(self, model_graph,
              sess,
              batch_size,
              feed_dict):
        """Runs the training step for the model
            Args:
                model_graph: a ModelGraph containing an update_step
                sess: the session to run in
        """
        assert isinstance(model_graph, ModelGraph)
        assert model_graph.update_step is not None

        sess.run(model_graph.update_step, feed_dict=feed_dict)
        #print(sess.run(model_graph.loss, feed_dict=feed_dict))

    def eval(self):
        pass

    def inference(self):
        pass

    def decode(self):
        pass

    def _get_infer_maximum_iterations(self, hparams, source_sequence_length):
        """Maximum decoding steps at inference time."""
        if hparams.tgt_max_len_infer:
          maximum_iterations = hparams.tgt_max_len_infer
        else:
          decoding_length_factor = 2.0
          max_encoder_length = tf.reduce_max(source_sequence_length)
          maximum_iterations = tf.to_int32(tf.round(
              tf.to_float(max_encoder_length) * decoding_length_factor))
        return maximum_iterations

    def build_graph(self,
                    iterator,
                    mode,
                    batch_size,
                    graph,
                    vars_save_scope=None):

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.scope = "train"
        if mode == tf.contrib.learn.ModeKeys.INFER:
            self.scope = "infer"
        if mode == tf.contrib.learn.ModeKeys.EVAL:
            self.scope = "eval"
        max_gradient_norm = self.hparams.max_gradient_norm
        initial_learning_rate = self.hparams.initial_learning_rate

        if mode != tf.contrib.learn.ModeKeys.INFER:
            (src, src_seq_len), (tgt_in, tgt_in_seq_len), (tgt_out, tgt_out_seq_len) = iterator.get_next()
        else:
            (src, src_seq_len) = iterator.get_next()
            tgt_in, tgt_in_seq_len, tgt_out, tgt_out_seq_len = None, None, None, None
        """
        train_feed_dict = {"batch_size:0": self.hparams.train_batch_size,
            "max_len:0": self.hparams.max_training_sequence_length,
            "src_dataset_file_name:0": self.hparams.train_src_dataset_file_name}#,
            #"tgt_dataset_file_name:0": self.hparams.train_tgt_dataset_file_name}
        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer, feed_dict=train_feed_dict)
            print(sess.run(src))
            exit(0)
        """
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            assert max_gradient_norm and initial_learning_rate

        encoder_outputs, encoder_state = self._build_encoder(src, src_seq_len)

        if mode == tf.contrib.learn.ModeKeys.INFER:
            maximum_iterations = self._get_infer_maximum_iterations(self.hparams, src_seq_len)
        else:
            maximum_iterations = tf.reduce_max(tgt_out_seq_len)

        logits, samples, final_context_state = self._build_decoder(mode,
            batch_size,
            encoder_state,
            maximum_iterations,
            tgt_in=tgt_in,
            tgt_in_seq_len=tgt_in_seq_len,
            tgt_out=tgt_out)

        if mode != tf.contrib.learn.ModeKeys.INFER:
            loss = self._build_loss(tgt_out,
                tgt_out_seq_len,
                logits,
                batch_size)
            tf.summary.scalar("loss", loss)
        else:
            loss = None

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            update_step = self._build_optimizer(loss,
                max_gradient_norm,
                initial_learning_rate)
        else:
            update_step = None
        """
        if mode == tf.contrib.learn.ModeKeys.INFER:

            infer_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)#, scope="infer")
            variable_mapping = {os.path.join("train", t.op.name): t for t in infer_variables}#{ os.path.join("train",t.op.name.split("/", 1)[1:][0]) : t for t in infer_variables}
            import pdb;pdb.set_trace()
            v = variable_mapping["train/embeddings_encoder"]
            del variable_mapping["train/embeddings_encoder"]
            variable_mapping["train/encoder_embeddings/embeddings_encoder"] = v
            v = variable_mapping["train/embeddings_decoder"]
            del variable_mapping["train/embeddings_decoder"]
            variable_mapping["train/decoder_embeddings/embeddings_decoder"] = v
            self.saver = tf.train.Saver(var_list=variable_mapping)
        elif mode == tf.contrib.learn.ModeKeys.EVAL:

            eval_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval")
            variable_mapping = { os.path.join("train",t.op.name.split("/", 1)[1:][0]) : t for t in eval_variables}
            self.saver = tf.train.Saver(var_list=variable_mapping)
        else:
        """

        if vars_save_scope is None:
            self.saver = tf.train.Saver()
        else:
            vars_save = []
            for scope in vars_save_scope:
                vars_save += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            vars_save_dict = {}
            for var_op in vars_save:
                vars_save_dict[var_op.op.name] = var_op
            lol = vars_save_dict["temporal_decoder_block/gated_linear_unit/kernel"]
            vars_save_dict['temporal_decoder_block/causal_gated_linear_unit/kernel'] = lol
            del vars_save_dict['temporal_decoder_block/gated_linear_unit/kernel']
            lol = vars_save_dict["temporal_decoder_block/gated_linear_unit/bias"]
            del vars_save_dict["temporal_decoder_block/gated_linear_unit/bias"]
            vars_save_dict['temporal_decoder_block/causal_gated_linear_unit/bias'] = lol

            lol = vars_save_dict["temporal_decoder_block_1/gated_linear_unit/kernel"]
            vars_save_dict['temporal_decoder_block_1/causal_gated_linear_unit/kernel'] = lol
            del vars_save_dict['temporal_decoder_block_1/gated_linear_unit/kernel']
            lol = vars_save_dict["temporal_decoder_block_1/gated_linear_unit/bias"]
            del vars_save_dict["temporal_decoder_block_1/gated_linear_unit/bias"]
            vars_save_dict['temporal_decoder_block_1/causal_gated_linear_unit/bias'] = lol


            lol = vars_save_dict["temporal_decoder_block_2/gated_linear_unit/kernel"]
            vars_save_dict['temporal_decoder_block_2/causal_gated_linear_unit/kernel'] = lol
            del vars_save_dict['temporal_decoder_block_2/gated_linear_unit/kernel']
            lol = vars_save_dict["temporal_decoder_block_2/gated_linear_unit/bias"]
            del vars_save_dict["temporal_decoder_block_2/gated_linear_unit/bias"]
            vars_save_dict['temporal_decoder_block_2/causal_gated_linear_unit/bias'] = lol

            lol = vars_save_dict["temporal_decoder_block_3/gated_linear_unit/kernel"]
            vars_save_dict['temporal_decoder_block_3/causal_gated_linear_unit/kernel'] = lol
            del vars_save_dict['temporal_decoder_block_3/gated_linear_unit/kernel']
            lol = vars_save_dict["temporal_decoder_block_3/gated_linear_unit/bias"]
            del vars_save_dict["temporal_decoder_block_3/gated_linear_unit/bias"]
            vars_save_dict['temporal_decoder_block_3/causal_gated_linear_unit/bias'] = lol


            print(vars_save_dict)
            self.saver = tf.train.Saver(vars_save_dict)

        return ModelGraph(logits=logits,
            samples=samples,
            loss=loss,
            num_units_per_cell=self.hidden_size,
            num_layers=self.num_layers,
            embeddings_size=self.embeddings_size,
            update_step=update_step,
            graph=graph,
            max_iter=maximum_iterations)

    def checkpoint_model(self, sess,
                         ckpt_path,
                         global_step):

        self.saver.save(sess, ckpt_path, global_step=global_step)

    def load_checkpointed_model(self, sess,
                                ckpt_path):
        self.saver.restore(sess, ckpt_path)

    def _build_embeddings_encoder(self):
        embeddings_encoder = tf.get_variable("embeddings_encoder",
            [self.src_vocab.size, self.embeddings_size], initializer=tf.truncated_normal_initializer(0, 0.1))

        return embeddings_encoder


    def _build_embeddings_decoder(self):
        assert self.tgt_vocab

        embeddings_decoder = tf.get_variable("embeddings_decoder",
            [self.tgt_vocab.size, self.embeddings_size], initializer=tf.truncated_normal_initializer(0, 0.1))

        return embeddings_decoder

    def _get_max_time(self, tensor):
        time_axis = 0 if self.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

    @abc.abstractmethod
    def _build_encoder(self, src,
                       src_seq_len):
        raise NotImplemented

    @abc.abstractmethod
    def _build_decoder(self, mode,
                       batch_size,
                       encoder_initial_state,
                       tgt_in=None,
                       tgt_in_seq_len=None,
                       tgt_out=None):
        raise NotImplemented

    @abc.abstractmethod
    def _build_proj_layer(self, tgt_vocab_size):
        raise NotImplemented

    def _get_learning_rate_warmup(self, hparams):
        """Get learning rate warmup."""
        warmup_steps = hparams.warmup_steps
        warmup_scheme = hparams.warmup_scheme

        # Apply inverse decay if global steps less than warmup steps.
        # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
        # When step < warmup_steps,
        #   learing_rate *= warmup_factor ** (warmup_steps - step)
        if warmup_scheme == "t2t":
          # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
          warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
          inv_decay = warmup_factor**(
              tf.to_float(warmup_steps - self.global_step))
        else:
          raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

        return tf.cond(
            self.global_step < hparams.warmup_steps,
            lambda: inv_decay * self.learning_rate,
            lambda: self.learning_rate,
            name="learning_rate_warump_cond")

    def _get_learning_rate_decay(self, hparams):
        """Get learning rate decay."""
        if hparams.decay_scheme in ["luong5", "luong10", "luong234"]:
          decay_factor = 0.5
          if hparams.decay_scheme == "luong5":
            start_decay_step = int(hparams.num_train_steps / 2)
            decay_times = 5
          elif hparams.decay_scheme == "luong10":
            start_decay_step = int(hparams.num_train_steps / 2)
            decay_times = 10
          elif hparams.decay_scheme == "luong234":
            start_decay_step = int(hparams.num_train_steps * 2 / 3)
            decay_times = 4
          remain_steps = hparams.num_train_steps - start_decay_step
          decay_steps = int(remain_steps / decay_times)
        elif not hparams.decay_scheme:  # no decay
          start_decay_step = hparams.num_train_steps
          decay_steps = 0
          decay_factor = 1.0
        elif hparams.decay_scheme:
          raise ValueError("Unknown decay scheme %s" % hparams.decay_scheme)

        return tf.cond(
            self.global_step < start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")


    def _build_optimizer(self,
                         loss,
                         max_gradient_norm,
                         initial_learning_rate):
        """Builds the Graph optimizer
            Args:
                loss: total loss tensor
                max_gradient_norm: clipping norm for gradient
                initial_learning_rate: learning rate to start at

            Returns:
                update_step: the update operation for the graph
        """

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)

        clipped_gradients, _ = tf.clip_by_global_norm(gradients,
            max_gradient_norm)


        self.learning_rate = tf.constant(initial_learning_rate)
        # warm-up
        self.learning_rate = self._get_learning_rate_warmup(self.hparams)
        # decay
        self.learning_rate = self._get_learning_rate_decay(self.hparams)

        if self.hparams.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.hparams.optimizer == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        update_step = optimizer.apply_gradients(zip(clipped_gradients, params),
            global_step = self.global_step)

        return update_step

    def _build_loss(self, tgt_out,
                    tgt_out_seq_len,
                    logits,
                    batch_size):
        """Builds Softmax cross entropy loss for decoder
            Args:
                tgt_out: tensor from iterator containing target labels
                tgt_out_seq_len: tensor from iterator containing target labels length
                batch_size: batch_size placeholder
                logits: output of projection layer from decoder

            Returns:
                loss: total loss tensor
        """

        if self.time_major:
            tgt_out = tf.transpose(tgt_out)

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tgt_out,
            logits=logits)

        target_weights = tf.sequence_mask(tgt_out_seq_len,
            self._get_max_time(tgt_out), dtype=tf.float32)

        if self.time_major:
            target_weights = tf.transpose(target_weights)

        loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(batch_size)
        tf.summary.scalar("loss", loss)
        return loss

class RNNNeuralTranslationModel(NeuralTranslationModel, abc.ABC):

    """Represents an RNN based Translation model seq2seq
    """

    def __init__(self, hparams,
                 src_vocab,
                 tgt_vocab):

        assert isinstance(hparams, RNNHyperParameters)

        self.hidden_size = hparams.num_units_per_cell
        self.num_layers = hparams.num_layers

        super(RNNNeuralTranslationModel, self).__init__(hparams,
            src_vocab, tgt_vocab)

    def _build_encoder(self, src,
                       src_seq_len):
        """Builds the encoder component of RNN model
            Args:
                src: tensor from dataset iterator representing input sentences
                src_seq_len: tensor from dataset iterator represneting input
                 sentence lengths

            Returns:
                encoder_outputs: output at each time step
                encoder_state: output of last hidden state
        """
        embeddings_encoder = self._build_embeddings_encoder()

        if self.time_major:
            src = tf.transpose(src)

        encoder_embeddings_inp = tf.nn.embedding_lookup(embeddings_encoder, src)



        if self.num_layers > 1:
            encoder_cell = self._build_multi_layer_cell(self.num_layers,
                self.num_units_per_cell)
        else:
            encoder_cell = self._build_cell(self.num_units_per_cell)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
            encoder_embeddings_inp, sequence_length=src_seq_len,
            time_major=self.time_major, scope="encoder",
            dtype=tf.float32)

        return encoder_outputs, encoder_state

    def _build_decoder(self,
                       mode,
                       batch_size,
                       encoder_initial_state,
                       maximum_iterations,
                       tgt_in=None,
                       tgt_in_seq_len=None,
                       tgt_out=None):


        if mode != tf.contrib.learn.ModeKeys.INFER:
            assert tgt_in is not None and tgt_in_seq_len is not None

        embeddings_decoder = self._build_embeddings_decoder()


        if self.num_layers > 1:
            decoder_cell = self._build_multi_layer_cell(self.num_layers,
                self.num_units_per_cell)
        else:
            decoder_cell = self._build_cell(self.num_units_per_cell)

        if mode != tf.contrib.learn.ModeKeys.INFER:
            decoder_emb_inp = tf.nn.embedding_lookup(
                embeddings_decoder, tf.transpose(tgt_in))

            helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_emb_inp, tgt_in_seq_len, time_major=self.time_major)

        else:
            start_tokens = tf.fill(tf.stack([batch_size]), self.tgt_vocab.sos_id_tensor)
            end_token = self.tgt_vocab.eos_id_tensor

            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings_decoder,
                start_tokens, end_token)

        projection_layer = self._build_proj_layer(self.tgt_vocab.size)

        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
            helper,
            encoder_initial_state,
            output_layer = projection_layer)

        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
            output_time_major=self.time_major,
            swap_memory=True,
            scope="decoder",
            maximum_iterations=maximum_iterations)

        logits = outputs.rnn_output
        samples = outputs.sample_id

        return logits, samples, final_context_state

    def _build_proj_layer(self, tgt_vocab_size):
        return tf.layers.Dense(tgt_vocab_size, use_bias=False)

    def _build_inference_helper(self):
        pass

    def _build_multi_layer_cell(self,
                                num_layers,
                                num_units_per_cell):
        """Generates multi-layer RNN cells
            Args:
                num_layers: number of layers
                num_units_per_cell: number of units per cell

            Returns:
                multi_layer_cell: the multi-layer cell
        """

        cells = [self._build_cell(num_units_per_cell, 0.2) for c in range(num_layers)]

        multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        return multi_layer_cell

    @abc.abstractmethod
    def _build_cell(self):
        raise NotImplemented

class TemporalCNNNeuralTranslationModel(NeuralTranslationModel):

    def __init__(self, hparams,
                 src_vocab,
                 tgt_vocab):

        assert isinstance(hparams, TemporalCNNHyperParameters)

        self.hidden_size = hparams.filters
        self.num_layers = hparams.num_layers

        super(TemporalCNNNeuralTranslationModel, self).__init__(hparams,
            src_vocab, tgt_vocab)

    def _build_encoder(self, src,
                       src_seq_len):

        embeddings_encoder = self._build_embeddings_encoder()
        encoder_embeddings_inp = tf.nn.embedding_lookup(embeddings_encoder, src)

        encoder = TemporalEncoder(num_layers=self.hparams.num_layers,
            kernel_size=self.hparams.kernel_size,
            filter_size=self.hparams.filters,
            embedding_size=self.hparams.embeddings_size)
        encoder = encoder(encoder_embeddings_inp)

        return None, encoder


    def _build_decoder(self,
                       mode,
                       batch_size,
                       encoder_initial_state,
                       maximum_iterations,
                       tgt_in=None,
                       tgt_in_seq_len=None,
                       tgt_out=None):

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            assert not (tgt_in is None or tgt_out is None)

            embeddings_decoder = self._build_embeddings_decoder()
            decoder_emb_inp = tf.nn.embedding_lookup(
                embeddings_decoder, tgt_in)
            decoder_emb_out = tf.nn.embedding_lookup(
                embeddings_decoder, tgt_out)
            """
            decoder = TemporalDecoder(num_layers=self.hparams.num_layers,
                kernel_size=self.hparams.kernel_size,
                filter_size=self.hparams.filters,
                embedding_size=self.hparams.embeddings_size)(decoder_emb_inp, encoder_initial_state, decoder_emb_out)
            """
            num_layers = self.hparams.num_layers
            kernel_size = self.hparams.kernel_size
            filters = self.hparams.filters
            embedding_size = self.hparams.embeddings_size

            inputs = decoder_emb_inp
            output = None
            for layer in range(num_layers):
                output =  TemporalDecoderBlock(kernel_size,
                    filters,
                    embedding_size,
                    inference=False,
                    padding="same",
                    kernel_initializer=None)(inputs, encoder_initial_state, decoder_emb_out)
                output = inputs

            projection = self._build_proj_layer(self.tgt_vocab.size)(output)

            return projection, tf.argmax(projection, axis=2), None

        elif mode == tf.contrib.learn.ModeKeys.INFER:
            maximum_iterations = tf.cast(maximum_iterations, tf.int64)
            num_layers = self.hparams.num_layers
            kernel_size = self.hparams.kernel_size
            filters = self.hparams.filters
            embeddings_size = self.hparams.embeddings_size

            sos_id_tensor = self.tgt_vocab.sos_id_tensor

            embeddings_decoder = self._build_embeddings_decoder()

            start_tensor = tf.constant([0] * (kernel_size))

            start_tensor = tf.tile(start_tensor, tf.stack([batch_size]))

            start_tensor = tf.reshape(start_tensor, shape=[-1, kernel_size])

            start_tensor = tf.cast(start_tensor, tf.int32)
            decoder_cell_in = start_tensor

            target = tf.fill(tf.stack([batch_size]), sos_id_tensor)
            target = tf.cast(target, tf.int32)
            target = tf.expand_dims(target, axis=1)

            sos_target =tf.fill(tf.stack([batch_size]), sos_id_tensor)
            sos_target = tf.cast(sos_target, tf.int32)
            sos_target = tf.expand_dims(sos_target, axis=1)

            zero_target = tf.fill(tf.stack([batch_size]), 0)
            zero_target = tf.cast(zero_target, tf.int32)
            zero_target = tf.expand_dims(zero_target, axis=1)

            counts = []
            cells = []
            cell_outputs = []
            for cell_num in range(num_layers):
                cell_output = tf.fill(tf.stack([batch_size, kernel_size, filters]), value=0.0)
                step_count = tf.Variable(tf.constant(0, shape=(), dtype=tf.int32), dtype=tf.int32)
                cell_fn = TemporalDecoderBlock(kernel_size,
                    filters,
                    embeddings_size,
                    inference=True,
                    padding="valid",
                    kernel_initializer=None)
                cells.append(cell_fn)
                cell_outputs.append(cell_output)
                counts.append(step_count)

            projection_fn = self._build_proj_layer(self.tgt_vocab.size)

            new_target = tf.fill(tf.stack([batch_size]), sos_id_tensor)
            new_target = tf.cast(new_target, tf.int32)
            new_target = tf.expand_dims(new_target, axis=1)

            target_count = tf.Variable(tf.constant(0, shape=(), dtype=tf.int32))

            #lol = tf.fill(tf.stack([batch_size,]), value="")
            def body(new_target, target, decoder_cell_in, cell_outputs, encoder_initial_state, embeddings_decoder, target_count, counts, kernel_size):

                    target = tf.cond(counts[0] < kernel_size - 1,
                        lambda: zero_target,
                        lambda: sos_target )
                    with tf.control_dependencies([target]):
                        target = tf.cond(counts[0] >= kernel_size,
                            lambda: new_target,
                            lambda: target)

                    #target = new_target
                        target = tf.Print(target, [target], "Current target: ")
                        decoder_cell_in = tf.concat([decoder_cell_in[:, 1:], target], axis=1)
                        decoder_cell_in = tf.Print(decoder_cell_in, [decoder_cell_in])
                        decoder_cell_in_embedding = tf.nn.embedding_lookup(embeddings_decoder, decoder_cell_in)
                        target_embedding = tf.nn.embedding_lookup(embeddings_decoder, target)
                        with tf.control_dependencies([decoder_cell_in_embedding, target_embedding]):

                            cell_outputs[0] = tf.concat([cell_outputs[0][:,1:,:], cells[0](decoder_cell_in_embedding, encoder_initial_state, target_embedding)], axis=1)
                            counts[0] += 1
                            cell_outputs[0] = tf.Print(cell_outputs[0], [cell_outputs[0]], "Cell output:")
                            #counts[0] +_= tf.cond(counts[0] > 1, lambda: counts[0] + 1, lambda: counts[0] + 1)
                            cell_num = 0

                            with tf.control_dependencies([cell_outputs[0], counts[0]]):
                            #for cell_num, objs  in enumerate(zip(cells[1:], cell_outputs[1:], cell_outputs[:-1], counts[:-1], counts[1:])):
                                for cell_num in range(len(cells)-1):
                                    cell_num += 1
                                    print(cell_num)
                                    #cell, cell_output, cell_input, count_in, count_out = objs
                                    cell = cells[cell_num]
                                    count_in = counts[cell_num-1]
                                    count_out = counts[cell_num]
                                    cell_input = cell_outputs[cell_num-1]
                                    cell_output = cell_outputs[cell_num]
                                    with tf.control_dependencies([cell_output, target, count_in]):
                                        cell_outputs[cell_num] = tf.cond(count_in >= kernel_size,
                                            lambda: tf.concat([cell_output[:,1:,:], cell(cell_input, encoder_initial_state, target_embedding)], axis=1),
                                            lambda: cell_output)
                                        with tf.control_dependencies([cell_outputs[cell_num], count_in]):
                                            counts[cell_num] = tf.cond(count_in >= kernel_size,
                                                lambda: count_out + 1,
                                                lambda: count_out)

                                with tf.control_dependencies([cell_outputs[-1], counts[-1]]):
                                    #counts[-1] = tf.Print(counts[-1], [counts[-1]], "Last Count")

                                    new_target = tf.cond(counts[-1]  > target_count,
                                        lambda: tf.cast(tf.argmax(projection_fn(tf.expand_dims(cell_outputs[-1][:,-1,:], axis=1)), axis=2), tf.int32),
                                        lambda: new_target)
                                    with tf.control_dependencies([new_target]):
                                            #counts[-1] = tf.Print(counts[-1], [counts[-1]], "Last Count before target")
                                            #target_count = tf.Print(target_count, [target_count], "Print target count is:")
                                            target_count = tf.cond(counts[-1] > target_count,
                                                lambda: target_count + 1,
                                                lambda: target_count)
                                    #with tf.control_dependencies([target_count]):
                                    #    new_target = tf.Print(new_target, [new_target])
                                    #lol = self.tgt_vocab.reverse_dict.lookup(tf.cast(new_target[:,0], tf.int64))
                                    #lol = tf.Print(lol, [lol])
                                    #with tf.control_dependencies([lol]):
                                    #    tf.no_op()
                                #decoder_cell_in = tf.Print(decoder_cell_in, [decoder_cell_in], "Decoder cell in is:")
                                #cell_outputs[0] = tf.Print(cell_outputs[0], [cell_outputs[0]], "First layer out is:")
                                #counts[0] = tf.Print(counts[0], [counts[0]], "Count 0")
                                #counts[1] = tf.Print(counts[1], [counts[1]], "Count 1")
                                #cells[0]._causal_gated_layer.weights[0] = tf.Print(cells[0]._causal_gated_layer.weights[0], [cells[0]._causal_gated_layer.weights[0]], "yo")
                        return  new_target, target, decoder_cell_in, cell_outputs, encoder_initial_state, embeddings_decoder, target_count, counts, kernel_size

            def condition(new_target, target, decoder_cell_in, cell_outputs, encoder_initial_state, embeddings_decoder, target_count, counts, kernel_size):
                return target_count < tf.cast(maximum_iterations, tf.int32)

            return None, tf.while_loop(condition, body, [ new_target, target, decoder_cell_in, cell_outputs, encoder_initial_state, embeddings_decoder, target_count, counts, kernel_size]), None

    def _build_inference_helper(self):
        pass

    def _build_proj_layer(self, tgt_vocab_size):
        return tf.layers.Conv1D(filters=tgt_vocab_size, kernel_size=1, activation=None, use_bias=False)

class LSTMNeuralTranslationModel(RNNNeuralTranslationModel):

    def _build_cell(self, num_units_per_cell, dropout_prob):
        """Build LSTM Cell
            Args:
                num_units_per_cell: number of units in a single cell
        """
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units_per_cell)
        if dropout_prob > 0.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell,
                input_keep_prob=(1.0 - dropout_prob))
        return cell
