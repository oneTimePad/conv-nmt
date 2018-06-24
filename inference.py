import model_helpers
import os
import tensorflow as tf

def inference(hparams, vars_save_scope):

    infer_scope = "infer"
    nt_model_infer = model_helpers.build_infer_model(hparams, vars_save_scope, scope="infer")
    model = nt_model_infer.model
    model_graph = nt_model_infer.model_graph
    iterator_tf = nt_model_infer.iterator_tf

    infer_feed_dict = {"batch_size:0": hparams.infer_batch_size,
        "max_len:0": hparams.max_training_sequence_length,
        "src_dataset_file_name:0": hparams.infer_src_dataset_file_name}

    tf.reset_default_graph()
    with model_graph.graph.as_default():
        with tf.Session() as sess:
           sess.run(tf.global_variables_initializer())
           model.load_checkpointed_model(sess, tf.train.latest_checkpoint(hparams.model_ckpt_dir))
           sess.run(tf.tables_initializer())
           sess.run(iterator_tf.initializer, feed_dict=infer_feed_dict)
           sample_ids = sess.run(model_graph.samples, feed_dict=infer_feed_dict)
           print(sess.run(model_graph.max_iter, feed_dict=infer_feed_dict))
           #import pdb;pdb.set_trace()
           #print(sample_ids)
           #for v in sample_ids:
           #         for t in v:
           #         print(nt_model_infer.tgt_vocab.reverse_dict[t], end=" ")
           #     print("\n")
