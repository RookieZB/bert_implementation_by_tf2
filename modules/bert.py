# -*- coding: utf-8 -*-

"""
Implementation of BERT, ALBERT, RoBERTa, ELECTRA
https://arxiv.org/abs/1810.04805
https://arxiv.org/abs/1909.11942
https://arxiv.org/abs/1907.11692
https://arxiv.org/abs/2003.10555

"""

import re
import json
import torch
import tensorflow as tf
import tensorflow.keras as keras


def w_initializing(param=0.02):
    return keras.initializers.TruncatedNormal(stddev=param)


def gelu_activating(x):
    return 0.5*x*(1.0+tf.math.erf(x/tf.math.sqrt(2.0)))


class Attention(keras.layers.Layer):
    def __init__(self, bname, lname, head, size, attdrop=0., drop=0., eps=1e-6, ninf=-1e4, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.head, self.size, self.dim, self.ninf = head, size, head*size, ninf
        self.wq = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+lname[0])
        self.wk = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+lname[1])
        self.wv = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+lname[2])
        self.dense = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+lname[3])
        self.norm = keras.layers.LayerNormalization(-1, eps, name=bname+lname[4])
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

    def propagating(self, x, mask=None, training=False, past=None):
        q1, k1, v1 = self.wq(x), self.wk(x), self.wv(x)
        k1 = k1 if past is None else tf.concat([past[0], k1], axis=-2)
        v1 = v1 if past is None else tf.concat([past[1], v1], axis=-2)
        m1 = mask if past is None else tf.concat([tf.zeros([tf.shape(x)[0], tf.shape(past[0])[1]], tf.int32), mask], 1)
        x1, a1 = self.calculating(q1, k1, v1, m1, training)
        x1 = tf.reshape(tf.transpose(x1, [0, 2, 1, 3]), [-1, tf.shape(x1)[2], self.dim])
        return self.norm(x+self.drop(self.dense(x1), training=training)), a1


class TransEncoder(keras.layers.Layer):
    def __init__(self, bname, lname, head, size, dff, attdrop=0., drop=0., act='relu', eps=1e-6, **kwargs):
        super(TransEncoder, self).__init__(**kwargs)
        self.att = Attention(bname, lname, head, size, attdrop, drop, eps)
        self.norm = keras.layers.LayerNormalization(-1, eps, name=bname+lname[7])
        self.dense1 = keras.layers.Dense(dff, act, True, w_initializing(), name=bname+lname[5])
        self.dense2 = keras.layers.Dense(int(head*size), None, True, w_initializing(), name=bname+lname[6])
        self.drop = keras.layers.Dropout(drop)

    def propagating(self, x, mask=None, training=False, past=None):
        x1, a1 = self.att.propagating(x, mask, training, past)
        return self.norm(x1+self.drop(self.dense2(self.dense1(x1)), training=training))


class Embedding(keras.layers.Layer):
    def __init__(self, bname, lname, voc, dim, maxlen=512, seg=2, drop=0., eps=1e-6, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.emb = self.add_weight(bname+lname[0], (voc, dim), None, w_initializing())
        self.posemb = self.add_weight(bname+lname[1], (maxlen, dim), None, w_initializing())
        self.segemb = self.add_weight(bname+lname[2], (seg, dim), None, w_initializing())
        self.norm = keras.layers.LayerNormalization(-1, eps, name=bname+lname[3])
        self.drop = keras.layers.Dropout(drop)

    def propagating(self, x, seg, pos, training=False):
        p1 = tf.slice(self.posemb, [0, 0], [tf.shape(x)[1], -1]) if pos is None else tf.gather(self.posemb, pos)
        return self.drop(self.norm(tf.gather(self.emb, x)+tf.gather(self.segemb, seg)+p1), training=training)


class BERT(keras.layers.Layer):
    def __init__(self, config, model='bert', mlm=False, fromtf=True, **kwargs):
        super(BERT, self).__init__(**kwargs)
        assert model in ['bert', 'roberta', 'albert', 'electra']
        self.head, self.share, self.mlm = model if model in ['bert', 'electra'] else 'bert', (model == 'albert'), mlm
        self.param = json.load(open(config)) if type(config) is str else config
        self.edim = self.param.get('embedding_size', self.param['hidden_size'])
        self.act = gelu_activating if self.param['hidden_act'] == 'gelu' else self.param['hidden_act']
        self.nc = ['cls/predictions/'+i1 for i1 in ['transform/dense', 'transform/LayerNorm', 'output_bias']]
        self.na = [
            '/attention/self/query',
            '/attention/self/key',
            '/attention/self/value',
            '/attention/output/dense',
            '/attention/output/LayerNorm',
            '/intermediate/dense',
            '/output/dense',
            '/output/LayerNorm']
        self.nb = [
            '/attention_1/self/query',
            '/attention_1/self/key',
            '/attention_1/self/value',
            '/attention_1/output/dense',
            '/LayerNorm',
            '/ffn_1/intermediate/dense',
            '/ffn_1/intermediate/output/dense',
            '/LayerNorm_1']
        self.embedding = Embedding(
            self.head+'/embeddings',
            ['/word_embeddings', '/position_embeddings', '/token_type_embeddings', '/LayerNorm'],
            self.param['vocab_size'],
            self.edim,
            self.param['max_position_embeddings'],
            self.param['type_vocab_size'],
            float(self.param['hidden_dropout_prob']))
        self.projection = keras.layers.Dense(
            self.param['hidden_size'],
            name=self.head+('/encoder/embedding_hidden_mapping_in' if self.share else '/embeddings_project'),
            kernel_initializer=w_initializing()) if self.edim != self.param['hidden_size'] else None
        self.encoder = [TransEncoder(
            self.head+('/encoder/transformer/group_0/inner_group_0' if self.share else '/encoder/layer_'+str(i1)),
            self.nb if self.share else self.na,
            self.param['num_attention_heads'],
            self.param['hidden_size']//self.param['num_attention_heads'],
            self.param['intermediate_size'],
            float(self.param['attention_probs_dropout_prob']),
            float(self.param['hidden_dropout_prob']),
            self.act) for i1 in range(1 if self.share else self.param['num_hidden_layers'])]
        self.dense = keras.layers.Dense(self.edim, self.act, True, w_initializing(), name=self.nc[0]) if mlm else None
        self.norm = keras.layers.LayerNormalization(-1, 1e-6, name=self.nc[1]) if mlm else None
        self.outbias = self.add_weight(self.nc[2], self.param['vocab_size'], None, 'zeros') if mlm else None
        self.fromtf, self.rpl = fromtf, {'': ''} if fromtf else {
            '/': '.', 'layer_': 'layer.', 'kernel': 'weight', '_embeddings': '_embeddings.weight', 'output_': ''}

    def loading(self, ckpt):
        _ = self.propagating(tf.ones((2, 2), tf.int32), tf.zeros((2, 2), tf.int32), tf.zeros((2, 2)), head=self.mlm)
        r1, l1 = re.compile('|'.join(map(re.escape, self.rpl))), torch.load(ckpt) if not self.fromtf else None
        n1 = [r1.sub((lambda x1: self.rpl[x1.group(0)]), i1.name[:-2]) for i1 in self.weights]
        keras.backend.batch_set_value(zip(self.weights, [tf.train.load_variable(ckpt, i1) if self.fromtf else l1[
            i1].numpy().T if 'weight' in i1 and 'embeddings.weight' not in i1 else l1[i1].numpy() for i1 in n1]))

    def propagating(self, x, seg, mask, pos=None, head=False, training=False, past=None):
        x1, x2 = self.embedding.propagating(x, seg, pos, training=training), None
        x1 = self.projection(x1) if self.projection else x1

        for i1 in range(self.param['num_hidden_layers']):
            x1 = self.encoder[0 if self.share else i1].propagating(x1, mask, training, past[i1] if past else None)

        x2 = self.norm(self.dense(x1)) if head else None
        x2 = tf.nn.bias_add(tf.matmul(x2, self.embedding.emb, transpose_b=True), self.outbias) if head else None
        return (x2, x1) if head else x1
