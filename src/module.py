import tensorflow as tf
from ops import *
from hparams import args

def wavenet(inputs, mels, weight_norm=False, scope=None, reuse=None):
    with tf.variable_scope(scope or 'wavenet', reuse=reuse):
        ret = 0
        x = myconv1d(inputs=inputs,
                     output_channel=args.wavnet_channels,
                     filter_size=1,
                     weight_norm=weight_norm,
                     scope='conv_in',
                     reuse=reuse)
        for i in range(args.wavenet_layers): 
            x = myconv1d(inputs=x,
                         output_channel=2*args.wavnet_channels,
                         filter_size=args.wavenet_filter_size,
                         dilations=2**i,
                         weight_norm=weight_norm,
                         scope='dila_conv_'+str(i+1),
                         reuse=reuse)
            c = myconv1d(inputs=mels,
                         output_channel=2*args.wavnet_channels,
                         filter_size=1,
                         weight_norm=weight_norm,
                         scope='conv_mel_'+str(i+1),
                         reuse=reuse)
            x = x + c
            x = tf.tanh(x[:, :args.wavnet_channels]) * tf.sigmoid(x[:, args.wavnet_channels:])
            if i != args.wavenet_layers - 1:
                res =  myconv1d(inputs=x,
                                output_channel=2*args.wavnet_channels,
                                filter_size=1,
                                weight_norm=weight_norm,
                                scope='conv_x_'+str(i+1),
                                reuse=reuse)
                x += res[:, : args.wavnet_channels]
                ret += res[:, args.wavnet_channels:]
            else:
                ret += myconv1d(inputs=x,
                                output_channel=args.wavnet_channels,
                                filter_size=1,
                                weight_norm=weight_norm,
                                scope='conv_x_'+str(i+1),
                                reuse=reuse)
        ret = myconv1d(inputs=ret,
                       output_channel=inputs.get_shape()[1]*2,
                       filter_size=1,
                       var_init=tf.zeros_initializer(),
                       scope='conv_out',
                       reuse=reuse)
        ret = tf.split(ret, num_or_size_splits=2, axis=1)
        return ret

def conv_afclayer(inputs, mels, reverse=False, weight_norm=False, scope=None, reuse=None):
    with tf.variable_scope(scope or 'conv_afclayer', reuse=reuse):
        if not reverse:
            inputs = inv1x1conv1d(inputs, reverse, 'invconv1x1', reuse)
            inputs, logdet = inputs
        a, b = tf.split(inputs, num_or_size_splits=2, axis=1)
        logs, t = wavenet(a, mels, weight_norm=args.use_weight_norm, scope='WN', reuse=reuse)
        if not reverse:
            logs = tf.minimum(logs, 8.)
        s = tf.exp(logs)
        if not reverse:
            b = s * b + t
            ret = tf.concat([a, b], axis=1)
            return ret, tf.reduce_sum(logs, axis=1), logdet
        b = (b - t) / (s + 1e-12)
        ret = tf.concat([a, b], axis=1)
        ret = inv1x1conv1d(ret, reverse, 'invconv1x1', reuse)
        return ret

class buff():
    def __init__(self, loss_names):
        self.loss_names = loss_names
        self.buffers = [0. for x in self.loss_names]
        self.count = [0 for x in self.loss_names]
        self.loss_string = "Epoch : %r Batch : %r / %r "
        for loss_name in loss_names:
            self.loss_string = self.loss_string + loss_name + " : %.6f "

    def put(self, x, index):
        assert len(x) == len(index)
        for y, idx in zip(x, index):
            self.buffers[idx] += y
            self.count[idx] += 1.

    def printout(self, prefix):
        losses = tuple(prefix + [x / c if c != 0 else 0 for x, c in zip(self.buffers, self.count)])
        self.buffers = [0. for x in self.buffers]
        self.count = [0 for x in self.buffers]
        print (self.loss_string %losses)

    def get(self, index):
        return self.buffers[index] / self.count[index] if self.count[index] != 0 else 0
