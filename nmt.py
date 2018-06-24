import tensorflow as tf
from inference import inference
from train import train
from hparams import RNNHyperParameters, TemporalCNNHyperParameters
import os




"""
hparams = RNNHyperParameters(num_steps=1000000,
                             train_batch_size=128,
                             infer_batch_size=1,
                             max_training_sequence_length=50,
                             num_units_per_cell=128,
                             num_layers=2,
                             embeddings_size=128,
                             max_gradient_norm=5,
                             initial_learning_rate=1.0,
                             model_name="lstm",
                             src_lang = "en",
                             train_src_dataset_file_name = "/home/lie/nmt/nmt/scripts/iwslt15/train.en",
                             src_vocab_file_name = "/home/lie/nmt/nmt/scripts/iwslt15/vocab.en",
                             tgt_lang = "vi",
                             train_tgt_dataset_file_name = "/home/lie/nmt/nmt/scripts/iwslt15/train.vi",
                             tgt_vocab_file_name = "/home/lie/nmt/nmt/scripts/iwslt15/vocab.vi",
                             model_ckpt_dir="/tmp/nmt_new_tf",
                             infer_src_dataset_file_name="/home/lie/nmt/nmt/scripts/iwslt15/infer3.en")
"""
num_layers = 4
vars_save_scope = ["temporal_encoder", "temporal_decoder", "embeddings_encoder", "embeddings_decoder", "conv1d"]
for i in range(num_layers - 1):
    vars_save_scope += ["temporal_decoder_" + str(i + 1)]

hparams = TemporalCNNHyperParameters(num_steps=10000000,
                             train_batch_size=64,
                             infer_batch_size=1,
                             max_training_sequence_length=50,
                             filters=512,
                             kernel_size=3,
                             num_layers=num_layers,
                             embeddings_size=512,
                             max_gradient_norm=100000000,
                             initial_learning_rate=0.25,
                             model_name="temporal",
                             src_lang = "en",
                             optimizer="adam",
                             train_src_dataset_file_name = "/home/lie/nmt/nmt/scripts/iwslt15/train.en",
                             src_vocab_file_name = "/home/lie/nmt/nmt/scripts/iwslt15/vocab.en",
                             tgt_lang = "vi",
                             train_tgt_dataset_file_name = "/home/lie/nmt/nmt/scripts/iwslt15/train.vi",
                             tgt_vocab_file_name = "/home/lie/nmt/nmt/scripts/iwslt15/vocab.vi",
                             model_ckpt_dir="/home/lie/nmt_new_test4",
                             infer_src_dataset_file_name="/home/lie/nmt/nmt/scripts/iwslt15/infer.en",
                             time_major=False)

#train(hparams)
inference(hparams, vars_save_scope)
