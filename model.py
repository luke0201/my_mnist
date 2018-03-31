import tensorflow as tf

conv = tf.layers.conv2d
fc = tf.layers.dense
bnorm = tf.layers.batch_normalization
dropout = tf.layers.dropout

relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
sigmoid = tf.sigmoid
tanh = tf.tanh
softmax = tf.nn.softmax

def classifier(X, training, apply_softmax=True):
    with tf.variable_scope('classifier'):
        fc1 = fc(X, 1024, name='hidden1')
        fc1 = relu(fc1)
        fc1 = dropout(fc1, 0.3, training=training)

        fc2 = fc(fc1, 512, name='hidden2')
        fc2 = relu(fc2)
        fc2 = dropout(fc2, 0.3, training=training)

        fc3 = fc(fc2, 256, name='hidden3')
        fc3 = relu(fc3)
        fc3 = dropout(fc3, 0.3, training=training)

        output = fc(fc3, 10, name='output')
        if apply_softmax:
            output = softmax(output)

    return output
