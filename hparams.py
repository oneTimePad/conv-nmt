import tensorflow as tf


"""Classes for model hyperparameters
"""



class RNNHyperParameters(tf.contrib.training.HParams):
    """RNN type Hyper Parameters object
    """

    def __init__(self, num_steps,
                 train_batch_size,
                 infer_batch_size,
                 max_training_sequence_length,
                 num_units_per_cell,
                 num_layers,
                 embeddings_size,
                 max_gradient_norm,
                 initial_learning_rate,
                 model_name,
                 src_lang,
                 train_src_dataset_file_name,
                 src_vocab_file_name,
                 tgt_lang,
                 train_tgt_dataset_file_name,
                 tgt_vocab_file_name,
                 infer_src_dataset_file_name,
                 time_major=True,
                 warmup_scheme="t2t",
                 warmup_steps=0,
                 decay_scheme="",
                 num_train_steps=12000,
                 optimizer="sgd",
                 tgt_max_len_infer=None,
                 model_ckpt_dir="",
                 ckpt_frequency=100):

        super(RNNHyperParameters, self).__init__(train_batch_size=train_batch_size,
            max_training_sequence_length=max_training_sequence_length,
            num_units_per_cell=num_units_per_cell,
            num_layers=num_layers,
            embeddings_size=embeddings_size,
            max_gradient_norm=max_gradient_norm,
            initial_learning_rate=initial_learning_rate,
            model_name=model_name,
            time_major=time_major,
            warmup_scheme="t2t",
            warmup_steps=0,
            decay_scheme="",
            num_train_steps=12000,
            optimizer="sgd",
            tgt_max_len_infer=None,
            src_lang=src_lang,
            train_src_dataset_file_name=train_src_dataset_file_name,
            src_vocab_file_name=src_vocab_file_name,
            tgt_lang=tgt_lang,
            train_tgt_dataset_file_name=train_tgt_dataset_file_name,
            tgt_vocab_file_name=tgt_vocab_file_name,
            infer_src_dataset_file_name=infer_src_dataset_file_name,
            model_ckpt_dir=model_ckpt_dir,
            infer_batch_size=infer_batch_size,
            num_steps=num_steps,
            ckpt_frequency=ckpt_frequency)

    def hparam_type(self):
        return "RNN"


class TemporalCNNHyperParameters(tf.contrib.training.HParams):

    def __init__(self, num_steps,
                 train_batch_size,
                 infer_batch_size,
                 max_training_sequence_length,
                 filters,
                 kernel_size,
                 num_layers,
                 embeddings_size,
                 max_gradient_norm,
                 initial_learning_rate,
                 model_name,
                 src_lang,
                 train_src_dataset_file_name,
                 src_vocab_file_name,
                 tgt_lang,
                 train_tgt_dataset_file_name,
                 tgt_vocab_file_name,
                 infer_src_dataset_file_name,
                 time_major=True,
                 warmup_scheme="t2t",
                 warmup_steps=0,
                 decay_scheme="",
                 num_train_steps=12000,
                 optimizer="sgd",
                 tgt_max_len_infer=None,
                 model_ckpt_dir="",
                 ckpt_frequency=100):

        super(TemporalCNNHyperParameters, self).__init__(train_batch_size=train_batch_size,
            max_training_sequence_length=max_training_sequence_length,
            filters=filters,
            num_layers=num_layers,
            kernel_size=kernel_size,
            embeddings_size=embeddings_size,
            max_gradient_norm=max_gradient_norm,
            initial_learning_rate=initial_learning_rate,
            model_name=model_name,
            time_major=time_major,
            warmup_scheme="t2t",
            warmup_steps=0,
            decay_scheme="",
            num_train_steps=12000,
            optimizer="sgd",
            tgt_max_len_infer=None,
            src_lang=src_lang,
            train_src_dataset_file_name=train_src_dataset_file_name,
            src_vocab_file_name=src_vocab_file_name,
            tgt_lang=tgt_lang,
            train_tgt_dataset_file_name=train_tgt_dataset_file_name,
            tgt_vocab_file_name=tgt_vocab_file_name,
            infer_src_dataset_file_name=infer_src_dataset_file_name,
            model_ckpt_dir=model_ckpt_dir,
            infer_batch_size=infer_batch_size,
            num_steps=num_steps,
            ckpt_frequency=ckpt_frequency)

    def hparam_type(self):
        return "TemporalCNN"
