# -*- coding: utf-8 -*-

"""
Simple MobileBERT Implementation (not completed)
https://github.com/google-research/google-research/tree/master/mobilebert

"""

import re
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import mymodels as mm


class Normalization(mm.Normalization):
    def __init__(self, fake=True, **kwargs):
        super(Normalization, self).__init__(**kwargs)
        self.fake = fake

    def call(self, x, **kwargs):
        if self.fake:
            return self.gamma*x+self.beta
        else:
            m1, v1 = tf.nn.moments(x, self.axis, None, True)
            return self.gamma*(x-m1)*tf.math.rsqrt(v1+self.epsilon)+self.beta


class Embedding(mm.Embedding):
    def __init__(self, edim, fake, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        b1, l1 = kwargs['bname'], kwargs['lname']
        self.emb = self.add_weight(b1+l1[0], (kwargs['voc'], edim), None, mm.w_initializing())
        self.norm = Normalization(fake, method='layer', eps=kwargs.get('eps', 1e-6), name=b1+l1[3])
        self.dense = keras.layers.Dense(kwargs['dim'], None, True, mm.w_initializing(), name=b1+l1[4])

    def propagating(self, x, seg=None, pos=None, training=False):
        e1 = tf.gather(self.emb, x)
        e1 = [tf.pad(e1[:, 1:], ((0, 0), (0, 1), (0, 0))), e1, tf.pad(e1[:, :-1], ((0, 0), (1, 0), (0, 0)))]
        e1 = self.dense(tf.concat(e1, -1))+self.segemb(seg)+tf.slice(self.posemb, [0, 0], [tf.shape(x)[1], -1])
        return self.drop(self.norm(e1), training=training)


class Attention(mm.Attention):
    def __init__(self, fake, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.fake, b1, l1 = fake, kwargs['bname'], kwargs['lname']
        self.norm = Normalization(fake, method='layer', eps=1e-6, name=b1+l1[4])

    def propagating(self, x, mask=None, past=None, training=False):
        x1, a1 = self.calculating(self.wq(x[1]), self.wk(x[2]), self.wv(x[3]), mask, False, training)
        x1 = tf.reshape(tf.transpose(x1, [0, 2, 1, 3]), [-1, tf.shape(x1)[2], self.dim])
        return self.norm(x[0]+self.drop(self.dense(x1), training=training)), a1


class TransEncoder(mm.TransEncoder):
    def __init__(self, dim, fake, neck, neckatt, neckshare, neckd, ff, **kwargs):
        super(TransEncoder, self).__init__(**kwargs)
        b1, l1, h1, ad1, d1 = kwargs['bname'], kwargs['lname'], kwargs['head'], kwargs['attdrop'], kwargs['drop']
        self.neck, self.neckatt, self.neckshare, self.dim = neck, neckatt, neckshare, (neckd if neck else dim)
        self.att = Attention(fake, bname=b1, lname=l1, head=h1, size=self.dim//h1, attdrop=ad1, drop=d1*(not neck))
        self.dense2 = keras.layers.Dense(self.dim, None, True, mm.w_initializing(), name=b1+l1[6])
        self.norm = Normalization(fake, method='layer', eps=1e-6, name=b1+l1[7])
        self.inp = keras.layers.Dense(neckd, None, True, mm.w_initializing(), name=b1+l1[8]) if neck else None
        self.inpnorm = Normalization(fake, method='layer', eps=1e-6, name=b1+l1[9]) if neck else None
        self.inpatt = keras.layers.Dense(neckd, None, True, mm.w_initializing(), name=b1+l1[10]) if neckshare else None
        self.inpattnorm = Normalization(fake, method='layer', eps=1e-6, name=b1+l1[11]) if neckshare else None
        self.outp = keras.layers.Dense(dim, None, True, mm.w_initializing(), name=b1+l1[12])
        self.outpnorm = Normalization(fake, method='layer', eps=1e-6, name=b1+l1[13]) if neckshare else None
        self.ffnorm = [Normalization(fake, method='layer', eps=1e-6, name=b1+l1[16][i1]) for i1 in range(ff-1)]
        self.ffint = [keras.layers.Dense(
            kwargs['dff'], kwargs['act'], True, mm.w_initializing(), name=b1+l1[14][i1]) for i1 in range(ff-1)]
        self.ffout = [keras.layers.Dense(
            self.dim, None, True, mm.w_initializing(), name=b1+l1[15][i1]) for i1 in range(ff-1)]

    def propagating(self, x, mask=None, past=None, training=False):
        x1 = self.inpnorm(self.inp(x)) if self.neck else x
        s1 = self.inpattnorm(self.inpatt(x)) if self.neckshare else None
        q1, k1, v1 = (x1, x1, x1) if self.neckatt else (s1, s1, x) if self.neckshare else (x, x, x)
        x1, a1 = self.att.propagating([x1, q1, k1, v1], mask, training=training)

        for i1 in range(len(self.ffint)):
            x1 = self.ffnorm[i1](x1+self.ffout[i1](self.ffint[i1](x1)))

        x2 = self.dense2(self.dense1(x1))
        x2 = self.norm(x1+x2) if self.neck else self.norm(x1+self.drop(x2, training=training))
        return self.outpnorm(x+self.drop(self.outp(x2), training=training)) if self.neck else x2


class MobileBERT(keras.layers.Layer):
    def __init__(self, config, mode='seq', **kwargs):
        super(MobileBERT, self).__init__(**kwargs)
        self.param, self.mode = json.load(open(config)) if type(config) is str else config, mode
        self.act = mm.gelu_activating if self.param['hidden_act'] == 'gelu' else self.param['hidden_act']
        self.fakenorm = self.param['normalization_type'] == 'no_norm'
        self.replacement = {'embeddings/embeddings': 'embeddings'}
        self.namel = ['cls/predictions/'+i1 for i1 in [
            'transform/dense',
            'transform/LayerNorm',
            'output_bias',
            'extra_output_weights']]
        self.namee = [
            '/word_embeddings',
            '/position_embeddings',
            '/token_type_embeddings',
            '/FakeLayerNorm' if self.fakenorm else '/LayerNorm',
            '/embedding_transformation']
        self.namea = [
            '/attention/self/query',
            '/attention/self/key',
            '/attention/self/value',
            '/attention/output/dense',
            '/attention/output/'+('FakeLayerNorm' if self.fakenorm else 'LayerNorm'),
            '/intermediate/dense',
            '/output/dense',
            '/output/'+('FakeLayerNorm' if self.fakenorm else 'LayerNorm'),
            '/bottleneck/input/dense',
            '/bottleneck/input/'+('FakeLayerNorm' if self.fakenorm else 'LayerNorm'),
            '/bottleneck/attention/dense',
            '/bottleneck/attention/'+('FakeLayerNorm' if self.fakenorm else 'LayerNorm'),
            '/output/bottleneck/dense',
            '/output/bottleneck/'+('FakeLayerNorm' if self.fakenorm else 'LayerNorm'),
            ['/ffn_layer_{}/intermediate/dense'.format(i1) for i1 in range(self.param['num_feedforward_networks']-1)],
            ['/ffn_layer_{}/output/dense'.format(i1) for i1 in range(self.param['num_feedforward_networks']-1)],
            [('/ffn_layer_{}/output/'+('FakeLayerNorm' if self.fakenorm else 'LayerNorm')).format(
                i1) for i1 in range(self.param['num_feedforward_networks']-1)]]
        self.embedding = Embedding(
            self.param['embedding_size'],
            self.fakenorm,
            bname='bert/embeddings',
            lname=self.namee,
            voc=self.param['vocab_size'],
            dim=self.param['hidden_size'],
            maxlen=self.param['max_position_embeddings'],
            seg=self.param['type_vocab_size'],
            drop=float(self.param['hidden_dropout_prob']))
        self.encoder = [TransEncoder(
            self.param['hidden_size'],
            self.fakenorm,
            self.param.get('use_bottleneck', False),
            self.param.get('use_bottleneck_attention', False),
            self.param.get('key_query_shared_bottleneck', False),
            self.param['intra_bottleneck_size'],
            self.param['num_feedforward_networks'],
            bname='bert/encoder/layer_'+str(i1),
            lname=self.namea,
            head=self.param['num_attention_heads'],
            size=self.param['hidden_size']//self.param['num_attention_heads'],
            dff=self.param['intermediate_size'],
            drop=float(self.param['attention_probs_dropout_prob']),
            attdrop=float(self.param['hidden_dropout_prob']),
            act=self.act) for i1 in range(self.param['num_hidden_layers'])]

        if self.mode == 'mlm':
            extra1 = self.param['hidden_size']-self.param['embedding_size']
            self.dense = keras.layers.Dense(self.param['hidden_size'], self.act, name=self.namel[0])
            self.norm = keras.layers.LayerNormalization(-1, 1e-6, name=self.namel[1])
            self.outbias = self.add_weight(self.namel[2], self.param['vocab_size'], None, 'zeros')
            self.extra = self.add_weight(
                self.namel[3], [self.param['vocab_size'], extra1], None, mm.w_initializing()) if extra1 > 0 else None

    def loading(self, ckpt):
        _ = self.propagating(tf.ones((2, 2), tf.int32), tf.zeros((2, 2)), tf.zeros((2, 2)))
        r1 = re.compile('|'.join(map(re.escape, self.replacement)))
        n1 = [r1.sub((lambda x1: self.replacement[x1.group(0)]), i1.name[:-2]) for i1 in self.weights]
        keras.backend.batch_set_value(zip(self.weights, [tf.train.load_variable(ckpt, i1) for i1 in n1]))

    def propagating(self, x, seg, mask, softmax=True, training=False):
        x1, x2 = self.embedding.propagating(x, seg, training=training), None

        for i1 in range(self.param['num_hidden_layers']):
            x1 = self.encoder[i1].propagating(x1, mask, training=training)

        if self.mode == 'mlm':
            e1 = self.embedding.emb if self.extra is None else tf.concat([self.embedding.emb, self.extra], -1)
            x2 = tf.nn.bias_add(tf.matmul(self.norm(self.dense(x1)), e1, transpose_b=True), self.outbias)
            x2 = tf.nn.softmax(x2) if softmax else x2

        return (x2, x1) if self.mode == 'mlm' else x1 if self.mode == 'seq' else x1[:, 0, :]


def mobilebert_testing(sentence=None):
    toke1 = mm.Tokenizer()
    toke1.loading('../models/mobilebert_base_en/vocab.txt')
    bert1 = MobileBERT('../models/mobilebert_base_en/bert_config.json', 'mlm')
    bert1.loading('../models/mobilebert_base_en/mobilebert_variables.ckpt')
    text1, segm1, mask1 = toke1.encoding('Have a good day.' if not sentence else sentence, pad=False)
    print(bert1.propagating(np.array([text1]), np.array([segm1]), np.array([mask1])))
