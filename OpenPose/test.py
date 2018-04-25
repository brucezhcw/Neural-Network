from datatools import batch_data
from model import FCN_Handnet
from modeltools import *
import tensorflow as tf
import tensorlayer as tl
import win_unicode_console, os
import matplotlib.pyplot as plt
import numpy as np
win_unicode_console.enable()

if __name__ == '__main__':
    test_data_path = '.\\ValidationData'

    sess = tf.InteractiveSession()

    train_op, img, label, stage, loss, acc = FCN_Handnet()

    saver = tf.train.Saver()
    saver.restore(sess, './Model/model.ckpt-' + str(findStep('Model')))

    m = 0
    loss_total = [0, 0, 0]
    acc_total = [np.zeros(6), np.zeros(6), np.zeros(6)]
    for img_batch, label_batch in batch_data(test_data_path, 16):
        m += img_batch.shape[0]
        loss_val, acc_val = sess.run([loss, acc], feed_dict={img : img_batch, label : label_batch})
        loss_total = [loss_total[i] + loss_val[i] for i in range(3)]
        acc_total = [acc_total[i] + acc_val[i] for i in range(3)]

    loss_total = [loss_total[i] / m for i in range(3)]
    acc_total = [acc_total[i] / m for i in range(3)]
    print("In %d Test:" % m)
    print("loss = ", loss_total)
    print("acc = ", acc_total)

    for img_batch, label_batch in batch_data(test_data_path, 1):
        stage_val = sess.run(stage, feed_dict={img : img_batch, label : label_batch})
        img_batch = img_batch[0,:,:,0]
        label_batch = label_batch[0]
        drawResult(img_batch, label_batch)     
        for _stage in stage_val:
            _stage = _stage[0]
            drawResult(img_batch, _stage)
        plt.show()
