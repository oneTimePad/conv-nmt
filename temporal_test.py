import tensorflow as tf




"""
constant_inp = tf.constant([[[1., 1., 1., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1., 1., 1.]]])





gu = GatedLinearUnit(filters=32, kernel_size=3)
output = gu(constant_inp)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    import pdb;pdb.set_trace()
    a = sess.run(output)
"""

input_sequence = tf.get_variable(shape=[1, 3, 4], initializer=tf.ones_initializer(), name="source")
conv_out = tf.layers.Conv1D(kernel_size=3, filters=3, activation=None)(input_sequence)
sequence = tf.constant([[  [1., 1., 1., 1.],
                          [1., 1., 1., 1.],
                          [1., 1., 1., 1.],
                          [2., 2., 2., 2.],
                          [2., 1., 1., 1.],
                          [2., 1., 1., 1.]]])

def body(conv_out, input_sequence, sequence, i):

    input_sequence = sequence[:,i:i+3,:]#.set_shape([1,3,4])
    #subset.set_shape([1, 3,4])
    return conv_out, input_sequence, sequence, i +1

def condition(conv_out, input_sequence, sequence, i):
    return i < 1

i = tf.Variable(tf.constant(0, shape=()))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = tf.while_loop(condition, body, [conv_out, input_sequence, sequence, i],
    shape_invariants=[tf.TensorShape([None, None,None]), tf.TensorShape([None,None,None]), tf.TensorShape([None,None,None]), tf.TensorShape(())])
    for r in result:
        print(r.eval())
        print("\n")
