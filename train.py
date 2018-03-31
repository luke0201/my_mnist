import os
import time
import numpy as np
import tensorflow as tf

import model
from data_loader import MNISTLoader

n_epoch = 100
batch_size = 200
log_dir = 'log'
checkpoint_dir = 'checkpoint'

X = tf.placeholder(tf.float32, [batch_size, 784], name='X')
y = tf.placeholder(tf.int32, [batch_size], name='y')

# Preprocess
X_preproc = tf.scalar_mul(2. / 255., X)
X_preproc = tf.add(X_preproc, tf.constant(-1., shape=X_preproc.shape))

logits = model.classifier(X_preproc, training=True, apply_softmax=False)

# Loss
loss = tf.losses.sparse_softmax_cross_entropy(y, logits, scope='loss')
tf.summary.scalar('Loss', loss)

# Evaluation metric
with tf.name_scope('evaluation'):
    corrects = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
    tf.summary.scalar('Accuracy', accuracy)

# Train ops
learning_rate = tf.get_variable(
        'learning_rate', [1], tf.float32,
        initializer=tf.constant_initializer(0.01, tf.float32), trainable=False)
decay_lr_op = learning_rate.assign(tf.scalar_mul(0.1, learning_rate))
optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = optimizer.minimize(loss)

def main():
    mnist = MNISTLoader()
    print('Dataset loaded\n')

    saver = tf.train.Saver()

    print('Start training\n')
    start_time = time.time()

    with tf.Session() as sess:
        print()

        tf.global_variables_initializer().run()

        with tf.summary.FileWriter(
                os.path.join(log_dir, str(start_time)),
                graph=sess.graph) as writer:

            iter_per_epoch = mnist.train.size // batch_size
            prev_acc_val = 0.
            for iteration in range(1, n_epoch * iter_per_epoch + 1):
                X_batch, y_batch = mnist.train.next_batch(batch_size)

                loss_val, acc_val, _ = sess.run(
                        [loss, accuracy, train_op],
                        feed_dict={X: X_batch, y: y_batch})

                if iteration % iter_per_epoch == 0:
                    cur_epoch = iteration // iter_per_epoch

                    # Print summary
                    print('Epoch:', cur_epoch)
                    print('- Loss:', loss_val)
                    print('- Accuracy:', acc_val)

                    summary = tf.summary.merge_all().eval(
                            feed_dict={X: X_batch, y: y_batch})
                    writer.add_summary(summary, global_step=cur_epoch)
                    print()

                    if iteration % 30 == 0:
                        print('Decay learning rate')
                        sess.run(decay_lr_op)
                        print()
                    prev_acc_val = acc_val

        saver.save(sess, os.path.join(checkpoint_dir, 'final-' + str(start_time)))

    print('Training done')
    end_time = time.time()
    print('Elapsed time: ', end_time - start_time)

if __name__ == '__main__':
    main()
