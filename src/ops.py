import tensorflow as tf
import numpy as np
from hparams import args
var_initializer = tf.random_normal_initializer(stddev=0.02)
magic = 'inter_wn_param'

def get_var(name,
            shape=None,
            dtype=tf.float32,
            initializer=None,
            trainable=True,
            weight_norm=False):
    if weight_norm:
        assert len(shape) == 4 #shuold be convolution
        g = tf.get_variable(name=name+magic+'_g', shape=[1, 1, 1, shape[-1]], dtype=dtype, initializer=initializer, trainable=trainable)
        v = tf.get_variable(name=name+magic+'_v', shape=shape, dtype=dtype, initializer=initializer, trainable=trainable)
        return g * tf.nn.l2_normalize(v, axis=[0, 1, 2])
    ret = tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable)
    return ret

def mydeconv2d(inputs,
               output_channel,
               filter_size=[1, 1],
               strides=[1, 1, 1, 1],
               scope=None,
               reuse=None):
    if type(filter_size) == int:
        filter_size = [filter_size, filter_size]
    if len(strides) == 2:
        strides = [1, 1] + strides
    with tf.variable_scope(scope or 'deconv', reuse=reuse):
        input_shape = inputs.get_shape().as_list()
        input_shape[2] = tf.shape(inputs)[2]
        batch_size = tf.shape(inputs)[0]
        input_channel = input_shape[1]
        weight_shape = [filter_size[0], filter_size[1], output_channel, input_channel]
        output_shape = [batch_size, output_channel, input_shape[2], input_shape[3]]
        output_shape = [a * b if a is not None else None for a, b in zip(output_shape, strides)]
        weights = get_var("weights", shape=weight_shape, initializer=var_initializer)
        biases = get_var("biases", shape=[output_channel], initializer=tf.zeros_initializer())
        deconv = tf.nn.conv2d_transpose(inputs, weights, output_shape=output_shape, strides=strides, data_format="NCHW")
        return tf.nn.bias_add(deconv, biases, data_format="NCHW")

def mydeconv1d(inputs,
               output_channel,
               filter_size=1,
               stride=1,
               scope=None,
               reuse=None):
    #example input : (batch_size, ch_size, width)
    filter_size = [filter_size, 1]
    strides = [1, 1, stride, 1]
    inputs = tf.expand_dims(inputs, -1)
    ret = mydeconv2d(inputs, output_channel, filter_size, strides, scope, reuse)
    return tf.squeeze(ret, -1)

def myconv2d(inputs,
             output_channel,
             filter_size=[1, 1],
             strides=[1, 1, 1, 1],
             padding='SAME',
             dilations=[1, 1, 1, 1],
             weight_norm=False,
             scope=None,
             reuse=None):
    if type(filter_size) == int:
        filter_size = [filter_size, filter_size]
    if len(strides) == 2:
        strides = [1, 1] + strides
    with tf.variable_scope(scope or 'conv2d', reuse=reuse):
        input_channel = inputs.get_shape().as_list()[1]
        weight_shape = [filter_size[0], filter_size[1], input_channel, output_channel]
        weights = get_var("weights", shape=weight_shape, initializer=var_initializer, weight_norm=weight_norm)
        biases = get_var("biases", shape=[output_channel], initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(inputs, weights, strides=strides, padding=padding, data_format="NCHW", dilations=dilations)
        return tf.nn.bias_add(conv, biases, data_format="NCHW")

def myconv1d(inputs,
             output_channel,
             filter_size=1,
             stride=1,
             padding='SAME',
             dilations=1,
             var_init=None,
             weight_norm=False,
             scope=None,
             reuse=None):
    #example input : (batch_size, ch_size, width)
    if var_init is not None:
        global var_initializer
        temp = var_initializer
        var_initializer = var_init
    filter_size = [filter_size, 1]
    strides = [1, 1, stride, 1]
    dilations = [1, 1, dilations, 1]
    inputs = tf.expand_dims(inputs, -1)
    ret = myconv2d(inputs, output_channel, filter_size, strides, padding, dilations, weight_norm, scope, reuse)
    if var_init is not None:
        var_initializer = temp
    return tf.squeeze(ret, -1)

def inv1x1conv2d(inputs, reverse=False, scope=None, reuse=None):
    with tf.variable_scope(scope or 'inv1x1_2d', reuse=reuse):
        w_shape = [inputs.get_shape()[1], inputs.get_shape()[1]]
        init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        init[:, 0] *= -1
        w = get_var('weights', initializer=init)
        w = tf.reshape(w, [1, 1] + w_shape)
        if not reverse:
            logdet = tf.cast(tf.log(tf.abs(tf.linalg.det(tf.cast(w, tf.float64)))), tf.float32)
            logdet *= tf.cast(tf.shape(inputs)[0] * tf.shape(inputs)[2] * tf.shape(inputs)[3], tf.float32)
            ret = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], 'SAME', data_format="NCHW")
            return ret, logdet
        w = tf.linalg.inv(w)
        ret = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], 'SAME', data_format="NCHW")
        return ret

def inv1x1conv1d(inputs, reverse=False, scope=None, reuse=None):
    inputs = tf.expand_dims(inputs, -1)
    if not reverse:
        ret, logdet = inv1x1conv2d(inputs, reverse, scope, reuse)
        return tf.squeeze(ret, -1), logdet
    ret = inv1x1conv2d(inputs, reverse, scope, reuse)
    return tf.squeeze(ret, -1)
