
import tensorflow as tf
import codecs
import os
import numpy as np


UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"


class NeuralTranslation(object):


    def __init__(self,
                 embedding_size,
                 num_units):
        self.vocab_tables = {}
        self.vocab_size = {}
        self.embedding_size = embedding_size
        self.num_units = num_units

    def load_vocab(self, vocab_file_name,
                         lang):

        vocab = []
        vocab_size = 0
        #with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file_name), "rb") as f:
        with tf.gfile.GFile(vocab_file_name) as f:
            for word in f:
                vocab.append(word.strip())
                vocab_size += 1

        if not EOS in vocab:
            vocab = [EOS] + vocab
        if not SOS in vocab:
            vocab = [SOS] + vocab
        if not UNK in vocab:
            vocab = [UNK] + vocab
        self.reverse_dictionaries = {}
        new_vocab_file_name = vocab_file_name + ".new"
        with tf.gfile.GFile(new_vocab_file_name, "wb") as f:
            self.reverse_dictionaries[lang] = {}
            i = 0
            for word in vocab:
                f.write("%s\n" % word)
                self.reverse_dictionaries[lang].update({i : word})
                i+=1

        self.vocab_tables.update({lang: tf.contrib.lookup.index_table_from_file(new_vocab_file_name, default_value = 0 )})
        self.vocab_size.update({lang: vocab_size})

    def generate_training_dataset(self,
                                  src_train_file,
                                  tgt_train_file,
                                  src_lang,
                                  tgt_lang,
                                  max_len,
                                  batch_size,
                                  reshuffle_each_iteration = True,
                                  output_buffer_size = 10):

        eos_id = tf.cast(self.vocab_tables[src_lang].lookup(tf.constant(EOS)), tf.int32)
        sos_id = tf.cast(self.vocab_tables[src_lang].lookup(tf.constant(SOS)), tf.int32)

        src_dataset = tf.data.TextLineDataset(src_train_file)
        tgt_dataset = tf.data.TextLineDataset(tgt_train_file)

        src_tgt_dataset = tf.data.Dataset.zip((src_dataset,tgt_dataset))

        src_tgt_dataset = src_tgt_dataset.shuffle(
            output_buffer_size, None, reshuffle_each_iteration)

        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt:
            (tf.string_split([src]).values, tf.string_split([tgt]).values)).prefetch(output_buffer_size)

        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt:
            (tf.cast(self.vocab_tables[src_lang].lookup(src), tf.int32), \
             tf.cast(self.vocab_tables[tgt_lang].lookup(tgt), tf.int32)) ).prefetch(output_buffer_size)



        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt:
            ((src, tf.size(src)),
            (tf.concat(([sos_id], tgt[1:]), 0), tf.size(tgt)),
            (tf.concat((tgt[:-1], [eos_id]), 0), tf.size(tgt)))).prefetch(output_buffer_size)

        dataset = src_tgt_dataset.padded_batch(
            batch_size,
            padded_shapes = ( (tf.TensorShape([None]), tf.TensorShape([])),
                           (tf.TensorShape([None]), tf.TensorShape([])),
                           (tf.TensorShape([None]), tf.TensorShape([]))),

            padding_values = ((eos_id, 0),
                             (eos_id, 0),
                             (eos_id, 0)))
        return dataset

    def get_infer_dataset(self,
                          src_train_file,
                          src_lang,
                          max_len,
                          batch_size):

        eos_id = tf.cast(self.vocab_tables[src_lang].lookup(tf.constant(EOS)), tf.int32)

        src_dataset = tf.data.TextLineDataset(src_train_file)

        src_dataset = src_dataset.map(lambda string: tf.string_split([string]).values)

        src_dataset = src_dataset.map(lambda string: tf.cast(self.vocab_tables[src_lang].lookup(string), tf.int32))

        src_dataset = src_dataset.map(lambda words: (words, tf.size(words)))

        #src_dataset = src_dataset.map(lambda words, size: (words, size ))

        dataset = src_dataset.padded_batch(
            batch_size,
            padded_shapes = (tf.TensorShape([None]), tf.TensorShape([])),

            padding_values = (eos_id, 0))

        return dataset

    def build_encoder(self, src,
                            src_len,
                            src_vocab_size,
                            embedding_size,
                            num_units):

        with tf.variable_scope("encoder") as scope:
            init_width = 0.5/embedding_size
            embedding_encoder = tf.get_variable("embedding_encoder",
            [src_vocab_size, embedding_size])
            #initializer = tf.random_uniform(
            #    [src_vocab_size, embedding_size], -init_width, init_width))

            encoder_emb_inp = tf.nn.embedding_lookup(
                embedding_encoder, tf.transpose(src))

            encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, encoder_emb_inp,
                sequence_length = src_len, time_major = True,
                scope = "encoder",
                dtype = tf.float32)
        return encoder_state

    def build_decoder(self, src_len,
                           tgt_vocab_size,
                           embedding_size,
                           num_units,
                           encoder_hidden_state,
                           mode,
                           tgt_in = None,
                           tgt_in_len = None,
                           tgt_lang = None,
                           batch_size = None):

        with tf.variable_scope("decoder") as scope:
            init_width = 0.5/embedding_size
            embedding_decoder = tf.get_variable("decoding_encoder",
            [tgt_vocab_size, embedding_size])
            #initializer = tf.random_uniform(
            #    [tgt_vocab_size, embedding_size], -init_width, init_width))



            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

            if mode != tf.contrib.learn.ModeKeys.INFER:
                decoder_emb_inp = tf.nn.embedding_lookup(
                    embedding_decoder,  tf.transpose(tgt_in))
                helper = tf.contrib.seq2seq.TrainingHelper(
                    decoder_emb_inp, tgt_in_len, time_major = True)
            else:
                print("INFER")

                sos_id = tf.cast(self.vocab_tables[tgt_lang].lookup(tf.constant(SOS)), tf.int32)
                eos_id = tf.cast(self.vocab_tables[tgt_lang].lookup(tf.constant(EOS)), tf.int32)
                import pdb;pdb.set_trace()
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding_decoder, tf.fill([batch_size], sos_id), eos_id)

                #helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                #    embedding_decoder, tf.fill([batch_size], sos_id), eos_id, softmax_temperature = 0.5)
            projection_layer = tf.layers.Dense(tgt_vocab_size, use_bias = False)


            decoder_cell = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                 helper,
                 encoder_hidden_state,
                 output_layer = projection_layer)

            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder_cell,
                 output_time_major = True,
                 swap_memory = True,
                 scope = "decoder",
                 maximum_iterations =  tf.round(tf.reduce_max(src_len) * 2))

            return outputs

    def get_loss(self, logits,
                       tgt_out,
                       tgt_out_len,
                       batch_size):
        import pdb;pdb.set_trace()
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = tf.transpose(tgt_out), logits = logits)

        target_weights = tf.sequence_mask(tgt_out_len,
            self.get_max_time(tf.transpose(tgt_out)), dtype = tf.float32)

        train_loss = (tf.reduce_sum(crossent * tf.transpose(target_weights)) / tf.to_float(batch_size))

        return train_loss

    def get_optimizer(self, train_loss,
                            max_gradient_norm,
                            learning_rate):

        params = tf.trainable_variables()
        gradients = tf.gradients(train_loss, params)

        clipped_gradients, _ = tf.clip_by_global_norm(gradients,
            max_gradient_norm)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        start_decay_step = int(10000 * 2 / 3)
        decay_times = 4
        self.global_step = tf.Variable(0, name = "global_step")
        remain_steps = 10000 - start_decay_step
        decay_steps = int(remain_steps / decay_times)
        decay_factor = 0.5
        learning_rate = tf.cond(
            self.global_step < start_decay_step,
            lambda: learning_rate,
            lambda: tf.train.exponential_decay(
                 learning_rate,
                 (self.global_step - start_decay_step),
                 decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        update_step = optimizer.apply_gradients(zip(clipped_gradients, params),  global_step = self.global_step)

        return update_step

    def get_max_time(self, tensor):
        time_axis = 0 # time_major
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

    def build_train_graph(self, src_train_file,
                          tgt_train_file,
                          src_lang,
                          tgt_lang,
                          max_len,
                          batch_size,
                          learning_rate,
                          max_gradient_norm):

        dataset = self.generate_training_dataset(src_train_file,
                                                  tgt_train_file,
                                                  src_lang,
                                                  tgt_lang,
                                                  max_len,
                                                  batch_size)
        iterator = dataset.make_initializable_iterator()
        ((src, src_len), (tgt_in, tgt_in_len), (tgt_out, tgt_out_len)) = iterator.get_next()

        encoder_state = self.build_encoder(src,
                                           src_len,
                                           self.vocab_size[src_lang],
                                           self.embedding_size,
                                           self.num_units)


        decoder_output = self.build_decoder(tgt_out_len,
                                            self.vocab_size[tgt_lang],
                                            self.embedding_size,
                                            self.num_units,
                                            encoder_state,
                                            mode = tf.contrib.learn.ModeKeys.TRAIN,
                                            tgt_in = tgt_in,
                                            tgt_in_len = tgt_in_len,
                                            tgt_lang = tgt_lang)

        loss = self.get_loss(decoder_output.rnn_output,
            tgt_out, tgt_out_len, batch_size)

        update_step = self.get_optimizer(loss,
            max_gradient_norm, learning_rate)

        return loss, update_step, iterator


    def build_infer_graph(self,
                          src_train_file,
                          src_lang,
                          tgt_lang,
                          batch_size):

        dataset = self.get_infer_dataset(src_train_file,
                               src_lang,
                               10,
                               batch_size)


        iterator = dataset.make_initializable_iterator()
        (src, src_len) = iterator.get_next()

        encoder_state = self.build_encoder(src, src_len,
                                           self.vocab_size[src_lang],
                                           self.embedding_size,
                                           self.num_units)

        decoder_output = self.build_decoder(src_len,
                                            self.vocab_size[tgt_lang],
                                            self.embedding_size,
                                            self.num_units,
                                            encoder_state,
                                            mode = tf.contrib.learn.ModeKeys.INFER,
                                            tgt_lang = tgt_lang,
                                            batch_size = batch_size)

        return decoder_output, iterator





n = NeuralTranslation(128, 128)
n.load_vocab("/home/lie/nmt/nmt/scripts/iwslt15/vocab.en", "en")
n.load_vocab("/home/lie/nmt/nmt/scripts/iwslt15/vocab.vi", "vi")



LOGDIR = "/tmp/nmt"

MODE = tf.contrib.learn.ModeKeys.INFER

if MODE == tf.contrib.learn.ModeKeys.TRAIN:
    loss, update_step, iterator = n.build_train_graph("/home/lie/nmt/nmt/scripts/iwslt15/train.en",
                                  "/home/lie/nmt/nmt/scripts/iwslt15/train.vi",
                                  "en",
                                  "vi",
                                  10,
                                  128,
                                  1.0,
                                  5)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(LOGDIR, sess.graph)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)
        for steps in range(10000):
            try:
                val, _ = sess.run([loss, update_step])
                print(val)
            except Exception:
                break
        saver.save(sess, os.path.join(LOGDIR, "model8.ckpt"))
elif MODE == tf.contrib.learn.ModeKeys.INFER:
    logits, iterator = n.build_infer_graph("/home/lie/nmt/nmt/scripts/iwslt15/tst2013.en",
                                           "en",
                                           "vi",
                                           2)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess,os.path.join(LOGDIR, "model8.ckpt"))
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)
        decode = sess.run(logits.sample_id)
        for batch in decode:
            for word in batch:
                print(n.reverse_dictionaries["vi"][word], end = " ")
            print("\n")
