import numpy as np
import struct
import matplotlib.pyplot as plt
import os
from PIL import Image
import sqlite3

# filename = 'train-images.idx3-ubyte'
filename = 't10k-images.idx3-ubyte'
binfile = open(filename, 'rb')
buf = binfile.read()

index = 0
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
index += struct.calcsize('IIII')
images = []
for i in range(numImages):
    imgVal = struct.unpack_from('>784B', buf, index)
    index += struct.calcsize('>784B')
    imgVal = list(imgVal)
    for j in range(len(imgVal)):
        if imgVal[j] > 1:
            imgVal[j] = 1

    images.append(imgVal)
arrX = np.array(images)

# 读取标签
#binFile = open('train-labels.idx1-ubyte', 'rb')
binFile = open('t10k-labels.idx1-ubyte','rb')
buf = binFile.read()
binFile.close()
index = 0
magic, numItems = struct.unpack_from('>II', buf, index)
index += struct.calcsize('>II')
labels = []
for x in range(numItems):
    im = struct.unpack_from('>1B', buf, index)
    index += struct.calcsize('>1B')
    labels.append(im[0])
arrY = np.array(labels)
print(np.shape(arrY))

open('mnist_test.db3', 'w')

conn = sqlite3.connect('mnist_test.db3')
cur = conn.cursor()
sql = "CREATE TABLE IF NOT EXISTS DATA(DATA BLOB, MEAN INTEGER)"
cur.execute(sql)

for i in range(10000):
	img = np.array(arrX[i])	
	sql = f"INSERT INTO DATA(DATA, MEAN) VALUES(\'{img}\',\'{arrY[i]}\')"
	cur.execute(sql)
	print(f'data 【{i+1}】')

conn.commit()