import tensorflow as tf
import collections
import model
from utils.iterator_utils import load_vocabs, Iterator
import os

__all__ = ["NTModel", "build_train_model"]

NTModel = collections.namedtuple("NTModel", ["src_vocab", "tgt_vocab", "iterator_tf", "model_graph", "model", "hparams", "mode"])

def _get_model_from_str_type(model_type):
    """Returns model class obj from string
        Args:
            model_type: string to select the model type
        Returns:
            class obj used to instantiate model
    """
    if model_type == "lstm":
        return model.LSTMNeuralTranslationModel

    if model_type == "temporal":
        return model.TemporalCNNNeuralTranslationModel

def build_train_model(hparams,
                      scope="train"):
    """Builds a training Seq2Seq model
        Args:
            hparams: a HParams object
            scope: scope of train model

        Returns:
            model: a NTModel tuple, representing a handle to the model
    """
    src_lang = hparams.src_lang
    src_vocab_file_name = hparams.src_vocab_file_name
    tgt_lang = hparams.tgt_lang
    tgt_vocab_file_name = hparams.tgt_vocab_file_name



    tf.reset_default_graph()

    train_graph = tf.Graph()
    with train_graph.as_default() as g:
        with tf.container(scope):
            src_vocab, tgt_vocab = load_vocabs(src_lang, src_vocab_file_name,
                tgt_lang, tgt_vocab_file_name)
            src_dataset_file_name = tf.placeholder(tf.string, name="src_dataset_file_name")
            tgt_dataset_file_name = tf.placeholder(tf.string, name="tgt_dataset_file_name")

            src_dataset = tf.data.TextLineDataset(src_dataset_file_name)
            tgt_dataset = tf.data.TextLineDataset(tgt_dataset_file_name)

            batch_size = tf.placeholder(tf.int64, name="batch_size")

            # maximum sequence length for training example
            max_len = tf.placeholder(tf.int64, name="max_len")

            iterator = Iterator(src_dataset, src_vocab,
                tgt_dataset, tgt_vocab, batch_size=batch_size, max_len=max_len)

            # actual TensorFlow Dataset Iterator
            iterator_tf = iterator.create_iterator()

            model_class = _get_model_from_str_type(hparams.model_name)

            model = model_class(hparams, src_vocab, tgt_vocab)

            model_graph = model.build_graph(iterator_tf,
                tf.contrib.learn.ModeKeys.TRAIN, batch_size, g)

            return NTModel(src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                iterator_tf=iterator_tf,
                model_graph=model_graph,
                model=model,
                hparams=hparams,
                mode=tf.contrib.learn.ModeKeys.TRAIN)

def build_eval_model(hparams,
                     scope="eval"):

    src_lang = hparams.src_lang
    src_vocab_file_name = hparams.src_vocab_file_name
    tgt_lang = hparams.tgt_lang
    tgt_vocab_file_name = hparams.tgt_vocab_file_name



    tf.reset_default_graph()

    eval_graph = tf.Graph()
    with eval_graph.as_default() as g:
        with tf.container(scope):
            src_vocab, tgt_vocab = load_vocabs(src_lang, src_vocab_file_name,
                tgt_lang, tgt_vocab_file_name)
            src_dataset_file_name = tf.placeholder(tf.string, name="src_dataset_file_name")
            tgt_dataset_file_name = tf.placeholder(tf.string, name="tgt_dataset_file_name")

            src_dataset = tf.data.TextLineDataset(src_dataset_file_name)
            tgt_dataset = tf.data.TextLineDataset(tgt_dataset_file_name)

            batch_size = tf.placeholder(tf.int64, name="batch_size")

            # maximum sequence length for training example
            max_len = tf.placeholder(tf.int64, name="max_len")

            iterator = Iterator(src_dataset, src_vocab,
                tgt_dataset, tgt_vocab, batch_size=batch_size, max_len=max_len)

            # actual TensorFlow Dataset Iterator
            iterator_tf = iterator.create_iterator(shuffle=False)

            model_class = _get_model_from_str_type(hparams.model_name)

            model = model_class(hparams, src_vocab, tgt_vocab)
            model_graph = model.build_graph(iterator_tf,
                tf.contrib.learn.ModeKeys.EVAL, batch_size, g)

            return NTModel(src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                iterator_tf=iterator_tf,
                model_graph=model_graph,
                model=model,
                hparams=hparams,
                mode=tf.contrib.learn.ModeKeys.EVAL)

def build_infer_model(hparams,
                      vars_save_scope,
                      scope="infer"):

    src_lang = hparams.src_lang
    src_vocab_file_name = hparams.src_vocab_file_name
    tgt_lang = hparams.tgt_lang
    tgt_vocab_file_name = hparams.tgt_vocab_file_name



    tf.reset_default_graph()

    infer_graph = tf.Graph()
    with infer_graph.as_default() as g:
        with tf.container(scope):
            src_vocab, tgt_vocab = load_vocabs(src_lang, src_vocab_file_name,
                tgt_lang, tgt_vocab_file_name)
            src_dataset_file_name = tf.placeholder(tf.string, name="src_dataset_file_name")

            src_dataset = tf.data.TextLineDataset(src_dataset_file_name)

            batch_size = tf.placeholder(tf.int64, shape=(), name="batch_size")

            # maximum sequence length for training example
            max_len = tf.placeholder(tf.int64, shape=(), name="max_len")

            iterator = Iterator(src_dataset, src_vocab,
                 tgt_vocab = tgt_vocab,
                 batch_size=batch_size, max_len=max_len)

            # actual TensorFlow Dataset Iterator
            iterator_tf = iterator.create_inference_iterator()
            model_class = _get_model_from_str_type(hparams.model_name)

            model = model_class(hparams, src_vocab, tgt_vocab)
            model_graph = model.build_graph(iterator_tf,
                tf.contrib.learn.ModeKeys.INFER, batch_size, g, vars_save_scope)

            return NTModel(src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                iterator_tf=iterator_tf,
                model_graph=model_graph,
                model=model,
                hparams=hparams,
                mode=tf.contrib.learn.ModeKeys.INFER)


def load_model(nt_model, sess, ckpt):
    model = nt_model.model
    model.load.checkpoint_model(sess, ckpt)
