# -*- coding: utf-8 -*-

"""
EfficientNet Implementation (not completed)
https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet

"""

import tensorflow as tf
import tensorflow.keras as keras


def w_initializing(scale=2., dist='truncated_normal'):
    return keras.initializers.VarianceScaling(scale, 'fan_out', dist)


def swish_activating(x):
    return x*keras.activations.sigmoid(x)


class MBConv(keras.layers.Layer):
    def __init__(self, ln, cin, cout, exp, kernel, stride, se, **kwargs):
        super(MBConv, self).__init__(**kwargs)
        self.cin, self.cout, self.exp, self.stride, self.se, conv1 = cin, cout, exp, stride, se, w_initializing
        self.conv, self.dep = keras.layers.Conv2D, keras.layers.DepthwiseConv2D
        name1 = [ln+i1 for i1 in ['conv2d', 'conv2d_1', 'depthwise_conv2d', 'se/conv2d', 'se/conv2d_1']]
        name2 = [ln+'tpu_batch_normalization'+i1 for i1 in ['', '_1', '_2']]

        if self.exp > 1:
            self.expc = self.conv(cin*exp, 1, 1, 'same', use_bias=False, kernel_initializer=conv1(), name=name1.pop(0))
            self.ebn = keras.layers.BatchNormalization(-1, 0.99, 0.001, name=name2.pop(0))

        if 0 < self.se <= 1:
            self.sea = self.conv(int(se*cin), 1, 1, 'same', kernel_initializer=conv1(), name=name1[-2])
            self.seb = self.conv(cin*exp, 1, 1, 'same', kernel_initializer=conv1(), name=name1[-1])

        self.dconv = self.dep(kernel, stride, 'same', use_bias=False, kernel_initializer=conv1(), name=name1[-3])
        self.dbn = keras.layers.BatchNormalization(-1, 0.99, 0.001, name=name2[0])
        self.pconv = self.conv(cout, 1, 1, 'same', use_bias=False, kernel_initializer=conv1(), name=name1[0])
        self.pbn = keras.layers.BatchNormalization(-1, 0.99, 0.001, name=name2[1])

    @staticmethod
    def dropping(x, prob, training):
        return x*tf.math.floor(prob+tf.random.uniform([tf.shape(x)[0], 1, 1, 1]))/prob if training else x

    def propagating(self, x, prob, training=False):
        x1 = swish_activating(self.ebn(self.expc(x), training=training)) if self.exp > 1 else x
        x1 = swish_activating(self.dbn(self.dconv(x1), training=training))
        x2 = swish_activating(self.sea(tf.reduce_mean(x1, [1, 2], keepdims=True))) if 0 < self.se <= 1 else None
        x1 = tf.nn.sigmoid(self.seb(x2))*x1 if 0 < self.se <= 1 else x1
        x1 = self.pbn(self.pconv(x1), training=training)
        return x+self.dropping(x1, prob, training) if self.stride == 1 and self.cin == self.cout else x1


class EfficientNet(keras.layers.Layer):
    def __init__(self, model, cls=1000, **kwargs):
        super(EfficientNet, self).__init__(**kwargs)
        self.model, self.count, self.divisor, self.enc, self.survival = model, 0, 8, [], 0.8
        self.conv, conv1 = keras.layers.Conv2D, w_initializing
        self.param = {
            'efficientnet-b0': (1.0, 1.0, 224, 0.2),
            'efficientnet-b1': (1.0, 1.1, 240, 0.2),
            'efficientnet-b2': (1.1, 1.2, 260, 0.3),
            'efficientnet-b3': (1.2, 1.4, 300, 0.3),
            'efficientnet-b4': (1.4, 1.8, 380, 0.4),
            'efficientnet-b5': (1.6, 2.2, 456, 0.4),
            'efficientnet-b6': (1.8, 2.6, 528, 0.5),
            'efficientnet-b7': (2.0, 3.1, 600, 0.5),
            'efficientnet-b8': (2.2, 3.6, 672, 0.5)}[model]
        self.args = [
            (1, 3, 1, 1, 32, 16, 0.25),
            (2, 3, 2, 6, 16, 24, 0.25),
            (2, 5, 2, 6, 24, 40, 0.25),
            (3, 3, 2, 6, 40, 80, 0.25),
            (3, 5, 1, 6, 80, 112, 0.25),
            (4, 5, 2, 6, 112, 192, 0.25),
            (1, 3, 1, 6, 192, 320, 0.25)]

        for i1 in self.args:
            for i2 in range(int(tf.math.ceil(self.param[1]*i1[0]).numpy())):
                self.count, n1 = self.count+1, '{}/blocks_{}/'.format(model, self.count)
                f1, f2 = self.calculating(i1[4], self.param[0]), self.calculating(i1[5], self.param[0])
                self.enc.append(MBConv(n1, f2 if i2 else f1, f2, i1[3], i1[1], 1 if i2 else i1[2], i1[6]))

        c1, c2 = self.calculating(32, self.param[0]), self.calculating(1280, self.param[0])
        self.stem = self.conv(c1, 3, 2, 'same', use_bias=False, kernel_initializer=conv1(), name=model+'/stem/conv2d/')
        self.sbn = keras.layers.BatchNormalization(-1, 0.99, 0.001, name=model+'/stem/tpu_batch_normalization/')
        self.head = self.conv(c2, 1, 1, 'same', use_bias=False, kernel_initializer=conv1(), name=model+'/head/conv2d/')
        self.hbn = keras.layers.BatchNormalization(-1, 0.99, 0.001, name=model+'/head/tpu_batch_normalization/')
        self.dense = keras.layers.Dense(cls, None, True, w_initializing(1./3., 'uniform'), name=model+'/head/dense/')
        self.pool, self.drop = keras.layers.GlobalAveragePooling2D(), keras.layers.Dropout(self.param[3])

    def calculating(self, channel, width):
        c1 = max(self.divisor, int(channel*width+self.divisor/2)//self.divisor*self.divisor)
        return int(c1+self.divisor if c1 < 0.9*channel*width else c1) if width else channel

    def loading(self, ckpt):
        _ = self.propagating(tf.ones((2, self.param[2], self.param[2], 3)))
        ckpt1 = [tf.train.load_variable(ckpt, i1.name[:-2]) for i1 in self.weights]
        keras.backend.batch_set_value(zip(self.weights, ckpt1))

    def propagating(self, x, training=False):
        x1 = swish_activating(self.sbn(self.stem(x), training=training))

        for i1 in range(len(self.enc)):
            x1 = self.enc[i1].propagating(x1, 1.0-(1.0-self.survival)*float(i1)/len(self.enc), training)

        x2 = swish_activating(self.hbn(self.head(x1), training=training))
        return self.dense(self.drop(self.pool(x2), training=training)), x2, x1
