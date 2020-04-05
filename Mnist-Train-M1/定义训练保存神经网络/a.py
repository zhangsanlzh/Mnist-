import tensorflow as tf
import struct
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

x = tf.placeholder("float", [None, 784], name='x')
W = tf.Variable(tf.zeros([784,10]), name = 'w')
b = tf.Variable(tf.zeros([10]), name = 'b')
y = tf.nn.softmax(tf.matmul(x,W) + b)

tf.add_to_collection('pred_network', y)

y_ = tf.placeholder("float", [None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.name_scope('loss'):
     loss = tf.reduce_mean(tf.square(y-y_))
     tf.summary.scalar('loss',loss)

init = tf.initialize_all_variables()
sess = tf.Session()
merge_summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

sess.run(init)
for i in range(10000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  print('第' + str(i+1) + '次训练：')

  if (i+1) % 250 == 0:
      summary_str = sess.run(merge_summary_op, feed_dict={x: batch_xs, y_: batch_ys})
      writer.add_summary(summary_str, i)

  if i == 9999:
      saver = tf.train.Saver()
      saver.save(sess, 'model.ckpt')