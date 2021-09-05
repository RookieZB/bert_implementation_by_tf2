# -*- coding: utf-8 -*-

"""
DPR Implementation (not completed)
https://github.com/facebookresearch/DPR

"""

import re
import torch
import tensorflow as tf
import tensorflow.keras as keras


def w_initializing(param=0.02):
    return keras.initializers.TruncatedNormal(stddev=param)


def gelu_activating(x):
    return 0.5*x*(1.0+tf.math.erf(x/tf.math.sqrt(2.0)))


class Attention(keras.layers.Layer):
    def __init__(self, bname, head, size, attdrop=0., drop=0., eps=1e-8, ninf=-1e4, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.head, self.size, self.dim, self.ninf = head, size, head*size, ninf
        self.wq = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+'self.query')
        self.wk = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+'self.key')
        self.wv = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+'self.value')
        self.dense = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+'output.dense')
        self.norm = keras.layers.LayerNormalization(-1, eps, name=bname+'output.LayerNorm')
        self.attdrop = keras.layers.Dropout(attdrop)
        self.drop = keras.layers.Dropout(drop)

    def transposing(self, x):
        return tf.transpose(tf.reshape(x, [-1, tf.shape(x)[1], self.head, self.size]), [0, 2, 1, 3])

    def masking(self, mask):
        return tf.cast(mask[:, tf.newaxis, tf.newaxis, :], tf.float32)*self.ninf

    def calculating(self, q, k, v, mask, training):
        a1 = tf.matmul(self.transposing(q), self.transposing(k), transpose_b=True)
        a1 = a1/tf.math.sqrt(tf.cast(self.size, tf.float32))
        a1 = tf.nn.softmax(a1+self.masking(mask) if mask is not None else a1, axis=-1)
        return tf.matmul(self.attdrop(a1, training=training), self.transposing(v)), a1

    def propagating(self, x, mask=None, training=False):
        x1, a1 = self.calculating(self.wq(x), self.wk(x), self.wv(x), mask, training)
        x1 = tf.reshape(tf.transpose(x1, [0, 2, 1, 3]), [-1, tf.shape(x1)[2], self.dim])
        return self.norm(x+self.drop(self.dense(x1), training=training)), a1


class TransEncoder(keras.layers.Layer):
    def __init__(self, bname, head, size, dff, attdrop=0., drop=0., eps=1e-8, act=None, **kwargs):
        super(TransEncoder, self).__init__(**kwargs)
        self.att = Attention(bname+'attention.', head, size, attdrop, drop, eps)
        self.norm = keras.layers.LayerNormalization(-1, eps, name=bname+'output.LayerNorm')
        self.dense1 = keras.layers.Dense(dff, act, True, w_initializing(), name=bname+'intermediate.dense')
        self.dense2 = keras.layers.Dense(int(head*size), None, True, w_initializing(), name=bname+'output.dense')
        self.drop = keras.layers.Dropout(drop)

    def propagating(self, x, mask=None, training=False):
        x1, a1 = self.att.propagating(x, mask, training)
        return self.norm(x1+self.drop(self.dense2(self.dense1(x1)), training=training))


class Embedding(keras.layers.Layer):
    def __init__(self, bname, voc, dim, maxlen=512, seg=2, drop=0., eps=1e-8, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.emb = self.add_weight(bname+'word_embeddings.weight', (voc, dim), None, w_initializing())
        self.posemb = self.add_weight(bname+'position_embeddings.weight', (maxlen, dim), None, w_initializing())
        self.segemb = keras.layers.Embedding(seg, dim, w_initializing(), name=bname+'token_type_embeddings')
        self.norm = keras.layers.LayerNormalization(-1, eps, name=bname+'LayerNorm')
        self.drop = keras.layers.Dropout(drop)

    def propagating(self, x, seg=None, training=False):
        e1 = tf.gather(self.emb, x)+self.segemb(seg)+tf.slice(self.posemb, [0, 0], [tf.shape(x)[1], -1])
        return self.drop(self.norm(e1), training=training)


class DPR(keras.layers.Layer):
    def __init__(self, model, voc=30522, seg=2, maxlen=512, lnum=12, head=12, dim=768, dff=3072, drop=0.1, **kwargs):
        super(DPR, self).__init__(**kwargs)
        self.rpl = {'/': '.', 's/embeddings': 's.weight', 'kernel': 'weight', 'gamma': 'weight', 'beta': 'bias'}
        self.embedding = Embedding(model+'.embeddings.', voc, dim, maxlen, seg, drop)
        self.pooler = keras.layers.Dense(dim, None, True, w_initializing(), name=model+'.pooler.dense')
        self.encoder = [TransEncoder(model+'.encoder.layer.{}.'.format(
            i1), head, dim//head, dff, drop, drop, act=gelu_activating) for i1 in range(lnum)]

    def loading(self, pth):
        _ = self.propagating(tf.ones((2, 2), tf.int32), tf.zeros((2, 2)), tf.zeros((2, 2)))
        r1, l1 = re.compile('|'.join(map(re.escape, self.rpl))), torch.load(pth)['model_dict']
        n1, t1 = [r1.sub((lambda x1: self.rpl[x1.group(0)]), i1.name[:-2]) for i1 in self.weights], []

        for i1 in n1:
            if 'weight' in i1 and 'embed' not in i1 and len(l1[i1].shape) == 2:
                t1.append(tf.transpose(l1[i1].cpu().numpy(), [1, 0]))
            else:
                t1.append(l1[i1].cpu().numpy())

        keras.backend.batch_set_value(zip(self.weights, t1))

    def propagating(self, x, seg, mask, training=False):
        x1 = self.embedding.propagating(x, seg, training=training)

        for i1 in range(len(self.encoder)):
            x1 = self.encoder[i1].propagating(x1, mask, training=training)

        return x1, self.pooler(x1[:, 0, :])


class Reader(DPR):
    def __init__(self, model, **kwargs):
        super(Reader, self).__init__(model=model, **kwargs)
        self.cls, self.exc = keras.layers.Dense(1, name='qa_classifier'), keras.layers.Dense(2, name='qa_outputs')

    def propagating(self, x, seg, mask, training=False):
        x1, p1 = super(Reader, self).propagating(x=x, seg=seg, mask=mask, training=training)
        return self.cls(x1[:, 0, :]), tf.unstack(self.exc(x1), axis=-1)
