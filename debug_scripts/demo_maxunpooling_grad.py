
import tensorflow as tf
import numpy as np



ksize   = [1, 2, 2, 1]
strides = [1, 2, 2, 1]

test_in = [0, 9,
           0, 0,
        
           9, 0, 
           0, 0,
        
           0, 9,
           0, 0,
        
           0, 0,
           9, 0]

test_out = [-2, 8,
            -4,-2,
        
             8,-2, 
            -2,-4,
             
            -2, 8,
            -4,-2,
             
            -2,-4,
             8,-2]



X = tf.placeholder("float")
Y = tf.placeholder("float")
factor = tf.Variable(tf.constant([1] * 2 * 2, shape=[2, 1, 1, 2], dtype=tf.float32))

maxp, maxp_arg = tf.nn.max_pool_with_argmax(X, ksize=ksize, strides=strides, padding="SAME")
scale          = tf.mul(maxp, factor)
maxup          = tf.nn.max_unpool(scale, X, maxp_arg, ksize=ksize, strides=strides, padding="SAME")
cost           = tf.sub(maxup, Y)

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)



with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    print ""
    print ""
    
    x = np.array(test_in, dtype=np.float).reshape((2, 2, 2, 2))
    y = np.array(test_out, dtype=np.float).reshape((2, 2, 2, 2))
    
    for i in range(5):
        
        # get cost and variable
        c = sess.run(cost, feed_dict={X:x, Y:y})
        v = sess.run(factor)
        
        print "Step " + str(i)
        print "-" * len("Step " + str(i))
        print "cost:"
        print c
        print "factor:"
        print v
        print ""
        print ""
    
        # optimize
        sess.run(optimizer, feed_dict={X:x, Y:y})
        
