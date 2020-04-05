import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python import pywrap_tensorflow
from PIL import Image

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    reader = pywrap_tensorflow.NewCheckpointReader('model.ckpt')
    # for key in reader.get_variable_to_shape_map():
    #     print('name: ',key)
    #     print(reader.get_tensor(key))

    # print('-'*40)
    # for i in reader.get_tensor('w'):
    #     print(i)
    # print(len(reader.get_tensor('w')))


    # input_data = tf.gfile.FastGFile('402_3.png','rb').read()
    # decode_data = tf.image.decode_png(input_data, 1)
    #
    # decode_data = tf.image.convert_image_dtype(decode_data, tf.float32)
    # image = np.array(tf.reshape(decode_data, [1, 28,28,1]))


    img = mpimg.imread('38_7.png')
    image = Image.fromarray(np.array(img))
    image = np.array(np.reshape(image, [-1,784]))

    w = sess.run('w:0')
    b = sess.run('b:0')

    result = np.add(np.matmul(image, w), b)
    print(np.argmax(result))