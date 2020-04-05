import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow
import sqlite3
from defs import *

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    reader = pywrap_tensorflow.NewCheckpointReader('model.ckpt')

    data, mean= GetNext(175, 1)

    w = sess.run('w:0')
    b = sess.run('b:0')

    result = np.add(np.matmul(data, w), b)
    print(np.argmax(result))