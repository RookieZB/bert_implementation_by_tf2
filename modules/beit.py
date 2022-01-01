# -*- coding: utf-8 -*-

"""
Implementation: BEiT (not completed)
References: https://arxiv.org/abs/2106.08254 & https://github.com/huggingface/transformers

"""

import re
import json
import torch
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def w_initializing(param=0.02):
    return keras.initializers.TruncatedNormal(stddev=param)


def gelu_activating(x):
    return 0.5*x*(1.0+tf.math.erf(x/tf.math.sqrt(2.0)))


class PositionBias(keras.layers.Layer):
    def __init__(self, bn, head, window):
        super(PositionBias, self).__init__()
        self.wd, self.dist = window, (2*window[0]-1)*(2*window[1]-1)+3
        self.pb = self.add_weight(bn+'relative_position_bias_table', (self.dist, head), None, tf.zeros_initializer())
        m1 = np.meshgrid(np.arange(window[1]), np.arange(window[0]))
        c1 = np.reshape(np.stack([m1[1], m1[0]]), [-1, window[0]*window[1]])
        c2 = np.transpose(c1[:, :, None]-c1[:, None, :], [1, 2, 0])
        c2[:, :, 0], c2[:, :, 1] = c2[:, :, 0]+(window[0]-1), c2[:, :, 1]+(window[1]-1)
        c2[:, :, 0] = c2[:, :, 0]*(2*window[1]-1)
        self.idx = np.zeros((window[0]*window[1]+1,)*2, dtype=np.int)
        self.idx[1:, 1:], self.idx[0, 0:] = np.sum(c2, -1), self.dist-3
        self.idx[0:, 0], self.idx[0, 0] = self.dist-2, self.dist-1

    def propagating(self):
        p1 = tf.gather(self.pb, tf.reshape(self.idx, [-1]))
        return tf.transpose(tf.reshape(p1, [self.wd[0]*self.wd[1]+1, self.wd[0]*self.wd[1]+1, -1]), [2, 0, 1])


class Attention(keras.layers.Layer):
    def __init__(self, bname, head, size, window, attdrop=0., drop=0., **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.head, self.size, self.dim = head, size, head*size
        self.posbias = PositionBias(bname+'attention.relative_position_bias.', head, window) if window else None
        self.wq = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+'attention.query')
        self.wk = keras.layers.Dense(self.dim, None, False, w_initializing(), name=bname+'attention.key')
        self.wv = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+'attention.value')
        self.dense = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+'output.dense')
        self.attdrop = keras.layers.Dropout(attdrop)
        self.drop = keras.layers.Dropout(drop)

    def transposing(self, x):
        return tf.transpose(tf.reshape(x, [-1, tf.shape(x)[1], self.head, self.size]), [0, 2, 1, 3])

    def calculating(self, q, k, v, posbias, training):
        a1 = tf.matmul(self.transposing(q), self.transposing(k), transpose_b=True)
        a1 = a1/tf.math.sqrt(tf.cast(self.size, tf.float32))
        a1 = a1+tf.expand_dims(self.posbias.propagating(), 0) if self.posbias is not None else a1
        a1 = tf.nn.softmax(a1+tf.expand_dims(posbias, 0) if posbias is not None else a1, axis=-1)
        return tf.matmul(self.attdrop(a1, training=training), self.transposing(v)), a1

    def propagating(self, x, posbias=None, training=False):
        x1, a1 = self.calculating(self.wq(x), self.wk(x), self.wv(x), posbias, training)
        x1 = tf.reshape(tf.transpose(x1, [0, 2, 1, 3]), [-1, tf.shape(x1)[2], self.dim])
        return self.drop(self.dense(x1), training=training), a1


class TransEncoder(keras.layers.Layer):
    def __init__(self, bname, head, size, dff, window, ini, pathdrop, attdrop, drop, act, eps=1e-6, **kwargs):
        super(TransEncoder, self).__init__(**kwargs)
        self.pdrop, init1 = pathdrop, keras.initializers.Constant(ini)
        self.att = Attention(bname+'attention.', head, size, window, attdrop, drop)
        self.norm1 = keras.layers.LayerNormalization(-1, eps, name=bname+'layernorm_before')
        self.norm2 = keras.layers.LayerNormalization(-1, eps, name=bname+'layernorm_after')
        self.dense1 = keras.layers.Dense(dff, act, True, w_initializing(), name=bname+'intermediate.dense')
        self.dense2 = keras.layers.Dense(int(head*size), None, True, w_initializing(), name=bname+'output.dense')
        self.lambda1 = self.add_weight(bname+'lambda_1', (int(head*size),), None, init1) if ini else None
        self.lambda2 = self.add_weight(bname+'lambda_2', (int(head*size),), None, init1) if ini else None

    def dropping(self, x, training):
        drop1 = tf.math.floor(self.pdrop+tf.random.uniform([tf.shape(x)[0], 1, 1]))
        return x*drop1/self.pdrop if training else x

    def propagating(self, x, posbias, training=False):
        x1, a1 = self.att.propagating(self.norm1(x), posbias, training)
        x1 = self.dropping(self.lambda1*x1 if self.lambda1 is not None else x1, training)+x
        x2 = self.dense2(self.dense1(self.norm2(x1)))
        return self.dropping(self.lambda2*x2 if self.lambda2 is not None else x2, training)+x1


class Embedding(keras.layers.Layer):
    def __init__(self, bname, psize, dim, drop=0., mask=False, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.conv = keras.layers.Conv2D(dim, psize, psize, name=bname+'patch_embeddings.projection')
        self.cls = self.add_weight(bname+'cls_token', (1, 1, dim), None, w_initializing())
        self.mask = self.add_weight(bname+'mask_token', (1, 1, dim), None, w_initializing()) if mask else None
        self.drop = keras.layers.Dropout(drop)

    def propagating(self, x, mask=None, training=False):
        c1 = self.cls*tf.ones([tf.shape(x)[0], 1, 1])
        c2 = tf.reshape(self.conv(x), [-1, (tf.shape(x)[1]//self.conv.strides[0])**2, self.conv.filters])
        mask1 = tf.expand_dims(mask, -1) if mask is not None else None
        c2 = c2*(1.-mask1)+self.mask*mask1 if mask is not None else c2
        return self.drop(tf.concat([c1, c2], 1), training=training)


class BEiT(keras.layers.Layer):
    def __init__(self, config, mode=None, **kwargs):
        super(BEiT, self).__init__(**kwargs)
        self.param, self.mode = json.load(open(config)) if type(config) is str else config, mode
        self.rpl = {'/': '.', 'kernel': 'weight', 'gamma': 'weight', 'beta': 'bias'}
        self.pb = PositionBias('beit.encoder.relative_position_bias.', self.param['num_attention_heads'], [self.param[
            'image_size']//self.param['patch_size']]*2) if self.param['use_shared_relative_position_bias'] else None
        self.emb = Embedding(
            'beit.embeddings.',
            self.param['patch_size'],
            self.param['hidden_size'],
            self.param['hidden_dropout_prob'],
            self.param['use_mask_token'])
        self.encoder = [TransEncoder(
            'beit.encoder.layer.{}.'.format(i1),
            self.param['num_attention_heads'],
            self.param['hidden_size']//self.param['num_attention_heads'],
            self.param['intermediate_size'],
            [self.param['image_size']//self.param['patch_size']]*2 if self.param['use_relative_position_bias'] else [],
            self.param['layer_scale_init_value'],
            self.param['drop_path_rate'],
            self.param['attention_probs_dropout_prob'],
            self.param['hidden_dropout_prob'],
            gelu_activating if self.param['hidden_act'] == 'gelu' else self.param['hidden_act'],
            self.param['layer_norm_eps']) for i1 in range(self.param['num_hidden_layers'])]

        if mode == 'mim':
            self.norm = keras.layers.LayerNormalization(-1, self.param['layer_norm_eps'], name='layernorm')
            self.head = keras.layers.Dense(self.param['vocab_size'], None, True, w_initializing(), name='lm_head')
        elif mode == 'cls':
            self.norm = keras.layers.LayerNormalization(-1, self.param['layer_norm_eps'], name='beit.pooler.layernorm')
            self.head = keras.layers.Dense(int(list(self.param['id2label'].keys())[-1])-1, name='classifier')

    def loading(self, pth):
        _ = self.propagating(tf.ones((2, self.param['image_size'], self.param['image_size'], 3)))
        r1, l1 = re.compile('|'.join(map(re.escape, self.rpl))), torch.load(pth)
        n1, t1 = [r1.sub((lambda x1: self.rpl[x1.group(0)]), i1.name[:-2]) for i1 in self.weights], []

        for i1 in n1:
            if len(l1[i1].shape) == 4:
                t1.append(tf.transpose(l1[i1].numpy(), [2, 3, 1, 0]))
            elif 'weight' in i1 and len(l1[i1].shape) == 2:
                t1.append(tf.transpose(l1[i1].numpy(), [1, 0]))
            else:
                t1.append(l1[i1].numpy())

        keras.backend.batch_set_value(zip(self.weights, t1))

    def propagating(self, image, training=False):
        x1, l1 = self.emb.propagating(image, None, training), []
        p1 = self.pb.propagating() if self.pb is not None else None

        for i1 in range(len(self.encoder)):
            x1 = self.encoder[i1].propagating(x1, p1, training)
            l1.append(x1)

        p1 = tf.reduce_mean(x1[:, 1:, :], 1) if self.mode == 'cls' else x1 if self.mode == 'mim' else None
        x2 = self.head(self.norm(p1)) if self.mode in ['mim', 'cls'] else None
        return x1, x2, l1
