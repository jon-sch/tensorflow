
import tensorflow as tf
import numpy as np



def max_pool(prev_layer, k=2):
    var, var_argmax = tf.nn.max_pool_with_argmax(prev_layer, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")
    return var, var_argmax


def max_unpool(prev_layer, prev_layer_pool, var_argmax, k=2):
    var_argmax_in_dummy = tf.ones_like(prev_layer_pool)
    var_argmax_dummy =    tf.ones_like(var_argmax)
    var = tf.nn.max_unpool(prev_layer, var_argmax_in_dummy, var_argmax_dummy, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")
    return var


def conv2d(prev_layer, name):
    w = weights[name]
    b = biases[name]
    
    var = tf.nn.conv2d(prev_layer, w, [1, 1, 1, 1], padding='SAME')
    var = tf.nn.bias_add(var, b)
    var = tf.nn.relu(var)
    
    return var


def conv2d_transpose(prev_layer, name, dropout_prob):
    w = weights[name]
    b = biases[name]
    
    dims = prev_layer.get_shape().dims[:3]
    dims.append(w.get_shape()[-2]) # adpot channels from weights (weight definition with switched channels for deconv!)
    out_shape = tf.TensorShape(dims)
    
    var = tf.nn.conv2d_transpose(prev_layer, w, out_shape, strides=[1, 1, 1, 1], padding="SAME")
    var = tf.nn.bias_add(var, b)
    
    if not dropout_prob is None:
        var = tf.nn.relu(var)
        var = tf.nn.dropout(var, dropout_prob)
    
    return var




weights = {
    "conv1":    tf.Variable(tf.random_normal([3, 3,  3, 16])),
    "conv2":    tf.Variable(tf.random_normal([3, 3, 16, 32])),
    "conv3":    tf.Variable(tf.random_normal([3, 3, 32, 32])),
    "deconv2":  tf.Variable(tf.random_normal([3, 3, 16, 32])),
    "deconv1":  tf.Variable(tf.random_normal([3, 3,  1, 16]))
}

biases = {
    "conv1":    tf.Variable(tf.random_normal([16])),
    "conv2":    tf.Variable(tf.random_normal([32])),
    "conv3":    tf.Variable(tf.random_normal([32])),
    "deconv2":  tf.Variable(tf.random_normal([16])),
    "deconv1":  tf.Variable(tf.random_normal([ 1]))
}



## build miniature network

x = tf.placeholder(tf.float32, [12, 20, 20, 3])
y = tf.placeholder(tf.float32, [12, 20, 20, 1])
p = tf.placeholder(tf.float32)


conv1                   = conv2d(x, "conv1")
maxp1, maxp1_arg_max    = max_pool(conv1)

conv2                   = conv2d(maxp1, "conv2")
maxp2, maxp2_arg_max    = max_pool(conv2)

conv3                   = conv2d(maxp2, "conv3")

maxup2                  = max_unpool(conv3, conv2, maxp1_arg_max)
deconv2                 = conv2d_transpose(maxup2, "deconv2", p)

maxup1                  = max_unpool(deconv2, conv1, maxp2_arg_max)
deconv1                 = conv2d_transpose(maxup1, "deconv1", None)



## Optimizing stuff

loss                    = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(deconv1, y))
optimizer               = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)



## Test Data
np.random.seed(123)

batch_x = np.where(np.random.rand(12, 20, 20, 3) > 0.5, 1.0, -1.0)
batch_y = np.where(np.random.rand(12, 20, 20, 1) > 0.5, 1.0,  0.0)
prob    = 0.5

with tf.Session() as session:
    tf.set_random_seed(123)
    
    print ""
    print ""
    print "init"
    session.run(tf.initialize_all_variables())
    print "init_done"
    
    for i in range(10):
        print "optimize"
        session.run(optimizer, feed_dict={x: batch_x, y: batch_y, p: prob})
        #print session.run(loss, feed_dict={x: batch_x, y: batch_y, p: prob})
        print "optimize done"
        print ""
        print ""
        
        print "step", i + 1, "done"
                                                  
    
    
    
