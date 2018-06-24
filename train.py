import tensorflow as tf
import model_helpers
import os

def sample_decode(model, model_graph, iterator_tf, scope, hparams):

    infer_feed_dict = {"batch_size:0": hparams.infer_batch_size,
         "max_len:0": hparams.max_training_sequence_length,
         "src_dataset_file_name:0": hparams.infer_src_dataset_file_name}

    tf.reset_default_graph()
    with model_graph.graph.as_default() as g:
        with tf.Session() as sess:
           model.load_checkpointed_model(sess, tf.train.latest_checkpoint(hparams.model_ckpt_dir))
           sess.run(iterator_tf.initializer, feed_dict=infer_feed_dict)
           sample_ids = sess.run(model_graph.samples).transpose()
           print(sample_ids)

def train(hparams):

    assert hparams.model_ckpt_dir is not None

    nt_model_train = model_helpers.build_train_model(hparams, scope="train")

    #nt_model_infer = model_helpers.build_infer_model(hparams, scope="infer")

    #nt_model_eval =  model_helpers.build_eval_model(hparams, scope="eval")

    model_train = nt_model_train.model
    model_train_graph = nt_model_train.model_graph

    #model_eval = nt_model_eval.model
    #model_eval_graph = nt_model_eval.model_graph

    #model_infer = nt_model_infer.model
    #model_infer_graph = nt_model_infer.model_graph

    train_feed_dict = {"batch_size:0": hparams.train_batch_size,
        "max_len:0": hparams.max_training_sequence_length,
        "src_dataset_file_name:0": hparams.train_src_dataset_file_name,
        "tgt_dataset_file_name:0": hparams.train_tgt_dataset_file_name}
    tf.reset_default_graph()




    with model_train_graph.graph.as_default() as g:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(hparams.model_ckpt_dir,g)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            latest_ckpt = tf.train.latest_checkpoint(hparams.model_ckpt_dir)

            if latest_ckpt:
                model_train.load_checkpointed_model(sess, latest_ckpt)
            else:
                sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(nt_model_train.iterator_tf.initializer, feed_dict=train_feed_dict)

            step = 0 if not latest_ckpt else int(latest_ckpt.split("-")[1]) + 1

            while step < hparams.num_steps:

                try:
                    model_train.train(model_train_graph, sess, hparams.train_batch_size, train_feed_dict)
                    summary = sess.run(merged, feed_dict=train_feed_dict)
                    writer.add_summary(summary, step)
                    step += 1
                    if step % hparams.ckpt_frequency == 0:
                        print("Checkpoint %d" % step)
                        #summary = sess.run(merged, feed_dict=train_feed_dict)
                        model_train.checkpoint_model(sess, os.path.join(hparams.model_ckpt_dir,"model.ckpt"), step)
                        #writer.add_summary(summary, step)
                except tf.errors.OutOfRangeError as e:
                    model_train.checkpoint_model(sess, os.path.join(hparams.model_ckpt_dir,"model.ckpt"), step)
                    sess.run(nt_model_train.iterator_tf.initializer, feed_dict=train_feed_dict)
    """
    model = nt_model_infer.model
    with tf.Session() as sess:
         model.load_checkpointed_model(sess, "/tmp/nmt_tf/model.ckpt-279700")
         sess.run(tf.tables_initializer())
         sess.run(nt_model_infer.iterator_tf.initializer, feed_dict={"inference/batch_size:0": 1, "src_dataset_file_name:0": "/home/lie/nmt/nmt/scripts/iwslt15/infer2.en"})
         #print(sess.run(nt_model_infer.iterator_tf.get_next(), feed_dict={"inference/batch_size:0": nt_model_infer.hparams.batch_size}))
         #exit(0)
         sample_ids = sess.run(nt_model_infer.model_graph.samples, feed_dict={"inference/batch_size:0": 1}).transpose()
         print(sample_ids)
         for v in sample_ids:
             for t in v:
                 print(nt_model_infer.tgt_vocab.reverse_dict[t], end=" ")
             print("\n")
    """
