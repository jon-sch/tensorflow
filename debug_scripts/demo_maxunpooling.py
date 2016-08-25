
import tensorflow as tf



ksize   = [1, 2, 2, 1]
strides = [1, 2, 2, 1]

data = [5, 8,
        1, 3,
        
        6, 4, 
        2, 1,
        
        1, 7,
        2, 1,
        
        4, 1,
        5, 2]

data_sub = [5,
            7,
            4,
            6]



x = tf.Variable(tf.constant(data, shape=[2, 2, 2, 2], dtype=tf.float32))

b = tf.Variable(tf.constant(data_sub, shape=[2, 1, 1, 2], dtype=tf.float32))

maxp, maxp_arg = tf.nn.max_pool_with_argmax(x, ksize=ksize, strides=strides, padding="SAME")

sub = tf.sub(maxp,b) # do some stuff in between

y = tf.nn.max_unpool(sub, x, maxp_arg, ksize=ksize, strides=strides, padding="SAME")

# Notes:
# max_unpool(input, input_of_corresponding_pooling_op, arg_max_of_corresponding_pooling_op, ...)
# 
# See also:
# tensorflow/core/kernels/maxpooling_op.cc
# tensorflow/core/ops/nn_ops.cc
# tensorflow/python/framework/ops.py


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    out = sess.run([x, maxp, maxp_arg, sub, y])
    
    print ""
    print ""
    
    for i, var_name in enumerate(["x", "maxp", "maxp_arg", "sub", "y"]):
        print var_name
        print out[i]
        print ""
    
    print ""
    print ""


# Used for debugging:
# http://stackoverflow.com/questions/218616/getting-method-parameter-names-in-python
    

