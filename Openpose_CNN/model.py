import numpy as np
import tensorflow as tf
import tensorlayer as tl

def argmax2D(tensor, imgsize = 48):
    _tensor = tf.reshape(tensor, [-1, imgsize ** 2, 6])
    index = tf.argmax(_tensor, 1)
    return index // imgsize, index % imgsize

def stage_acc(stage, label_x, label_y, imgsize = 48, acc_range = 3):
    stage_x, stage_y = argmax2D(stage, imgsize)
    correct = tf.sqrt(tf.cast((label_x - stage_x) ** 2 + (label_y - stage_y) ** 2, 'float')) <= acc_range
    return tf.reduce_sum(tf.cast(correct, 'float'), 0)

def FCN_Handnet(imgsize = 48, acc_range = 3):
    img = tf.placeholder(tf.float32, shape=(None, imgsize, imgsize, 1), name='img')
    label = tf.placeholder(tf.float32, shape=(None, imgsize, imgsize, 6), name='fingers')
    m = img.shape[0]

    feature_network = tl.layers.InputLayer(img, name='input')
    feature_network = tl.layers.Conv2d(feature_network, 32, (9,9), (1,1), act=tf.nn.relu, name='conv1')
    feature_network = tl.layers.Conv2d(feature_network, 64, (7,7), (1,1), act=tf.nn.relu, name='conv2')
    feature_network = tl.layers.Conv2d(feature_network, 64, (5,5), (1,1), act=tf.nn.relu, name='conv3')
    feature_network = tl.layers.Conv2d(feature_network, 128, (3,3), (1,1), act=tf.nn.relu, name='conv4')
    features = feature_network.outputs #21
    feature_network.print_layers
    feature_network.print_params

    stage1 = tl.layers.InputLayer(features, name='stage1_input')
    stage1 = tl.layers.Conv2d(stage1, 64, (7,7), (1,1), act=tf.nn.relu, name='stage1_conv1')
    stage1 = tl.layers.Conv2d(stage1, 32, (7,7), (1,1), act=tf.nn.relu, name='stage1_conv2')
    stage1 = tl.layers.Conv2d(stage1, 6, (1,1), (1,1), act=tf.nn.relu, name='stage1_conv3')
    stage1_output = stage1.outputs #33
    stage1.print_layers
    stage1.print_params

    stage2 = tl.layers.InputLayer(tf.concat([features, stage1_output], 3), name='stage2_input')
    stage2 = tl.layers.Conv2d(stage2, 64, (7,7), (1,1), act=tf.nn.relu, name='stage2_conv1')
    stage2 = tl.layers.Conv2d(stage2, 32, (7,7), (1,1), act=tf.nn.relu, name='stage2_conv2')
    stage2 = tl.layers.Conv2d(stage2, 6, (1,1), (1,1), act=tf.nn.relu, name='stage2_conv3')
    stage2_output = stage2.outputs #45
    stage2.print_layers
    stage2.print_params

    stage3 = tl.layers.InputLayer(tf.concat([features, stage2_output], 3), name='stage3_input')
    stage3 = tl.layers.Conv2d(stage3, 64, (7,7), (1,1), act=tf.nn.relu, name='stage3_conv1')
    stage3 = tl.layers.Conv2d(stage3, 32, (7,7), (1,1), act=tf.nn.relu, name='stage3_conv2')
    stage3 = tl.layers.Conv2d(stage3, 6, (1,1), (1,1), act=tf.nn.relu, name='stage3_conv3')
    stage3_output = stage3.outputs #57
    stage3.print_layers
    stage3.print_params

    stage1_loss = tf.nn.l2_loss(stage1_output - label)
    stage2_loss = tf.nn.l2_loss(stage2_output - label)
    stage3_loss = tf.nn.l2_loss(stage3_output - label)
    total_loss = stage1_loss + stage2_loss + stage3_loss
    train_op = tf.train.AdamOptimizer(1e-4).minimize(total_loss)

    label_x, label_y = argmax2D(label, imgsize)
    stage1_acc = stage_acc(stage1_output, label_x, label_y, imgsize, acc_range)
    stage2_acc = stage_acc(stage2_output, label_x, label_y, imgsize, acc_range)
    stage3_acc = stage_acc(stage3_output, label_x, label_y, imgsize, acc_range)

    return train_op, img, label, features, [stage1_output, stage2_output, stage3_output], [stage1_loss, stage2_loss, stage3_loss], [stage1_acc, stage2_acc, stage3_acc]

    

    