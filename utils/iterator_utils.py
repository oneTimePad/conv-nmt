import tensorflow as tf
import collections

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"

__all__ = ["Vocab", "Iterator", "load_vocabs"]

Vocab = collections.namedtuple("Vocab", ("lang", "table", "size", "reverse_dict", "sos_id_tensor", "eos_id_tensor"))

def _load_vocab(vocab_file_name, language):
    """Loads the vocabulary to a TensorFlow vocab table
        Args:
            vocab_file_name: vocab file string

        Returns:
            Vocab: Vocab named table holding the vocabulary attributes
    """
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

    reverse_dictionary = {}
    new_vocab_file_name = vocab_file_name + ".new"
    with tf.gfile.GFile(new_vocab_file_name, "wb") as f:
        reverse_dictionary = {}
        i = 0
        for word in vocab:
            f.write("%s\n" % word)
            reverse_dictionary.update({i : word})
            i+=1

    vocab_table = tf.contrib.lookup.index_table_from_file(new_vocab_file_name, default_value=0)
    reverse_table = tf.contrib.lookup.index_to_string_table_from_file(new_vocab_file_name, default_value="<unk>")
    eos_id_tensor = tf.cast(vocab_table.lookup(tf.constant(EOS)), tf.int32)
    sos_id_tensor = tf.cast(vocab_table.lookup(tf.constant(SOS)), tf.int32)

    return Vocab(lang=language,
                 table=vocab_table,
                 size=vocab_size,
                 reverse_dict=reverse_table,
                 sos_id_tensor=sos_id_tensor,
                 eos_id_tensor=eos_id_tensor)

def load_vocabs(src_lang,
                src_vocab_file_name,
                tgt_lang,
                tgt_vocab_file_name):
    """Constructs the Vocab tuples for both the src
    and tgt languages from the vocabulary files
        Args:
            src_lang: two letter language abbreviation
            src_vocab_file_name: file name for source vocab (contains each word line by line)
            tgt_lang: two letter lanaguage abbreviation
            tgt_vocab_file_name: file name for target vocab

        Returns:
            src_vocab: namedtuple for source vocab
            tgt_vocab: namedtuple for target vocab
    """

    src_vocab = _load_vocab(src_vocab_file_name, src_lang)
    tgt_vocab = _load_vocab(tgt_vocab_file_name, tgt_lang)

    return src_vocab, tgt_vocab

class Iterator(object):


    def __init__(self, src_dataset,
                       src_vocab,
                       tgt_dataset = None,
                       tgt_vocab = None,
                       batch_size = None,
                       max_len = None):
        """Constructs and Iterator for the given Model

        Note: batch size and datasets can be placeholders
        """
        assert batch_size is not None and max_len is not None

        self.src_dataset = src_dataset
        self.src_vocab = src_vocab

        self.tgt_dataset = tgt_dataset
        self.tgt_vocab = tgt_vocab

        self.batch_size = batch_size

        self.max_len = max_len

    def create_iterator(self,
                        shuffle=True,
                        reshuffle_each_iteration=True,
                        output_buffer_size=10,
                        num_parallel_calls=4):

        """Constructs the Training/Eval Pipeline iterator
            Args:
                reshuffle_each_iteration : whether to reshuffle the data each iteration
                output_buffer_size: prefetch output buffer

            Returns:
                iterator: iterator for full training/eval data pipeline
        """

        src_tgt_dataset = tf.data.Dataset.zip((self.src_dataset, self.tgt_dataset))
        if shuffle:
            src_tgt_dataset = src_tgt_dataset.shuffle(
                output_buffer_size, None, reshuffle_each_iteration)

        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt:
            (tf.string_split([src]).values, tf.string_split([tgt]).values),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)


        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt:
            (src[:self.max_len], tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt:
            (src, tgt[:self.max_len]),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt:
            (tf.cast(self.src_vocab.table.lookup(src), tf.int32),
             tf.cast(self.tgt_vocab.table.lookup(tgt), tf.int32)),
             num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt:
            ((src, tf.size(src)),
            (tf.concat(([self.tgt_vocab.sos_id_tensor], tgt), 0), tf.size(tgt)+1),
            (tf.concat((tgt, [self.tgt_vocab.eos_id_tensor]), 0), tf.size(tgt)+1)),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

        dataset = src_tgt_dataset.padded_batch(
            self.batch_size,
            padded_shapes = ( (tf.TensorShape([None]), tf.TensorShape([])),
                           (tf.TensorShape([None]), tf.TensorShape([])),
                           (tf.TensorShape([None]), tf.TensorShape([]))),

            padding_values = ((self.src_vocab.eos_id_tensor, 0),
                             (self.tgt_vocab.eos_id_tensor, 0),
                             (self.tgt_vocab.eos_id_tensor, 0)))

        iterator = dataset.make_initializable_iterator()

        return iterator

    def create_inference_iterator(self):
        """Constructs the inference tensor

            Return:
                iterator: iterator for full inference pipeline
        """

        src_dataset = self.src_dataset.map(lambda string: tf.string_split([string]).values)

        src_dataset = src_dataset.map(lambda string: tf.cast(self.src_vocab.table.lookup(string), tf.int32))

        src_dataset = src_dataset.map(lambda words: (words, tf.size(words)))

        dataset = src_dataset.padded_batch(
            self.batch_size,
            padded_shapes = (tf.TensorShape([None]), tf.TensorShape([])),

            padding_values = (self.src_vocab.eos_id_tensor, 0))


        iterator = dataset.make_initializable_iterator()

        return iterator
