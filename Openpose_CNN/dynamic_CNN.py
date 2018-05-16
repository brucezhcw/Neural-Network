from datatools import batch_data
from model import FCN_Handnet
from modeltools import *
import tensorflow as tf
import tensorlayer as tl
import win_unicode_console, os
import matplotlib.pyplot as plt
import numpy as np
win_unicode_console.enable()

def CNN_caculater():
  test_data_path = 'E:/Neural-Network/openpose-z/data/h5data/'
  train_op, img, label, features, stage, loss, acc = FCN_Handnet(acc_range=3)
  sess = tf.InteractiveSession()
  saver = tf.train.Saver()
  saver.restore(sess, 'E:/Neural-Network/openpose-z/Model/model.ckpt-260000')

  for img_batch, label_batch in batch_data(test_data_path, 1):
    stage_val = sess.run(stage, feed_dict={img : img_batch, label : label_batch})
    img_batch = img_batch[0,:,:,0] 
    drawResult(img_batch, stage_val[2][0])
    plt.show()
    break
if __name__ == '__main__':
    CNN_caculater()