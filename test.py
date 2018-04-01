import os

import tensorflow as tf

import model

batch_size = 200

input_dir = 'input'
dnn_checkpoint_path = os.path.join('checkpoint', 'final-1522507992.0969088')

with tf.name_scope('inference'):
    input_img = tf.placeholder(tf.string, [], name='input_img')
    X = tf.image.decode_png(input_img, channels=1)
    X = tf.reshape(X, [1, 784])
    logits = model.classifier(X, training=False)
    y = tf.argmax(logits, axis=0)
    confidence = tf.maximum(logits)

d_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classifier')
saver = tf.train.Saver(var_list=_var)

with tf.Session() as sess:
    print()

    saver.restore(sess, dnn_checkpoint_path)
    print('Model loaded\n')

    for filename in os.listdir(input_dir):
        full_path = os.path.join(input_dir, filename)
        with open(full_path, 'rb') as f:
            img = f.read()

        y_val, conf_val = sess.run([y, confidence], feed_dict={input_img: img})
        print('{}: {} ({:.2f})'.format(filename, y_val, conf_val))
