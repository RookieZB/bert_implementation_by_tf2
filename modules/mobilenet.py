# -*- coding: utf-8 -*-

"""
MobileNet V3
https://arxiv.org/abs/1905.02244

"""

import re
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def hswish_activating(x):
    return x*tf.nn.relu6(x+np.float32(3))*np.float32(1./6.)


def hsigmoid_activating(x):
    return tf.nn.relu6(x+3.)*0.16667


class Bottleneck(keras.layers.Layer):
    def __init__(self, ln, size, stride, cin, cout, exp, caxis, act, squ=True, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.div, self.cin, self.filter, self.stride, self.squeeze, self.exp = 8, cin, cout, stride, squ, int(cin*exp)
        self.act = {'hswish': hswish_activating, 'relu': tf.nn.relu}[act]
        self.conv1 = keras.layers.Conv2D(self.exp, 1, 1, 'same', use_bias=False, name=ln+'expand/')
        self.conv2 = keras.layers.DepthwiseConv2D(size, stride, 'same', use_bias=False, name=ln+'depthwise/')
        self.conv3 = keras.layers.Conv2D(cout, 1, 1, 'same', use_bias=False, name=ln+'project/')
        self.norm1 = keras.layers.BatchNormalization(caxis, 0.999, name=ln+'expand/BatchNorm/')
        self.norm2 = keras.layers.BatchNormalization(caxis, 0.999, name=ln+'depthwise/BatchNorm/')
        self.norm3 = keras.layers.BatchNormalization(caxis, 0.999, name=ln+'project/BatchNorm/')
        self.s1 = keras.layers.Conv2D(self.dividing(self.exp/4), 1, activation='relu', name=ln+'squeeze_excite/Conv/')
        self.s2 = keras.layers.Conv2D(self.exp, 1, activation=hsigmoid_activating, name=ln+'squeeze_excite/Conv_1/')
        self.pool = keras.layers.GlobalAveragePooling2D()

    def dividing(self, v, minv=None):
        valu1 = max(self.div if not minv else minv, int(v+self.div/2)//self.div*self.div)
        return valu1+self.div if valu1 < 0.9*v else valu1

    def propagating(self, x, training):
        x1 = self.act(self.norm1(self.conv1(x), training=training)) if self.cin != self.exp else x
        x1 = self.act(self.norm2(self.conv2(x1), training=training))
        h1 = tf.reshape(self.pool(x1), [-1, 1, 1, self.exp]) if self.squeeze else None
        x1 = x1*self.s2(self.s1(h1)) if self.squeeze else x1
        x1 = self.norm3(self.conv3(x1), training=training)
        return x+x1 if self.cin == self.filter and self.stride == 1 else x1


class MobileNet(keras.layers.Layer):
    def __init__(self, alpha, caxis, cate, act='hswish', param=None, **kwargs):
        super(MobileNet, self).__init__(**kwargs)
        self.act = hswish_activating if act == 'hswish' else act
        self.replacement = {'kernel': 'weights', 'bias': 'biases'}
        self.conv1 = keras.layers.Conv2D(16, 3, 2, 'same', use_bias=False, name='MobilenetV3/Conv/')
        self.conv2 = keras.layers.Conv2D(576, 1, use_bias=False, name='MobilenetV3/Conv_1/')
        self.conv3 = keras.layers.Conv2D(1024, 1, use_bias=False, name='MobilenetV3/Conv_2/')
        self.conv4 = keras.layers.Conv2D(cate, 1, use_bias=False, name='MobilenetV3/Logits/Conv2d_1c_1x1')
        self.norm1 = keras.layers.BatchNormalization(caxis, 0.999, 1e-3, name='MobilenetV3/Conv/'+'BatchNorm/')
        self.norm2 = keras.layers.BatchNormalization(caxis, 0.999, 1e-3, name='MobilenetV3/Conv_1/'+'BatchNorm/')
        self.pool = keras.layers.GlobalAveragePooling2D()
        self.param = [
            (16,),
            (16, 3, 2, 1, 'relu', True),
            (24, 3, 2, 72/16, 'relu', False),
            (24, 3, 1, 88/24, 'relu', False),
            (40, 5, 2, 4, 'hswish', True),
            (40, 5, 1, 6, 'hswish', True),
            (40, 5, 1, 6, 'hswish', True),
            (48, 5, 1, 3, 'hswish', True),
            (48, 5, 1, 3, 'hswish', True),
            (96, 5, 2, 6, 'hswish', True),
            (96, 5, 1, 6, 'hswish', True),
            (96, 5, 1, 6, 'hswish', True)] if not param else param
        self.encoder = [Bottleneck(
            'MobilenetV3/expanded_conv_'+str(i1-1)+'/' if i1 > 1 else 'MobilenetV3/expanded_conv/',
            self.param[i1][1],
            self.param[i1][2],
            self.param[i1-1][0]*alpha,
            self.param[i1][0]*alpha,
            self.param[i1][3],
            caxis, self.param[i1][4], self.param[i1][5]) for i1 in range(1, len(self.param))]

    def loading(self, ckpt):
        _ = self.propagating(tf.ones((2, 224, 224, 3)))
        r1 = re.compile('|'.join(map(re.escape, self.replacement)))
        n1 = [r1.sub((lambda x1: self.replacement[x1.group(0)]), i1.name[:-2]) for i1 in self.weights]
        keras.backend.batch_set_value(zip(self.weights, [tf.train.load_variable(ckpt, i1) for i1 in n1]))

    def propagating(self, x, training=False, softmax=True):
        x1 = self.act(self.norm1(self.conv1(x), training=training))

        for i1 in self.encoder:
            x1 = i1.propagating(x1, training)

        x1 = self.act(self.norm2(self.conv2(x1), training=training))
        x2 = self.act(self.conv3(tf.reshape(self.pool(x1), [-1, 1, 1, x1.shape[-1]])))
        return tf.nn.softmax(self.conv4(x2)) if softmax else self.conv4(x2), x1
