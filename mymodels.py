# -*- coding: utf-8 -*-

"""
Title: My Implementation of Models by TensorFlow
Author: Chris Chen
Date: 2020/02/02
Update: 2020/09/25

"""

import os
import warnings
import re
import regex
import unicodedata
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


"""Functions"""


def w_initializing(param=0.02):
    """Truncated normal initializing"""
    return keras.initializers.TruncatedNormal(stddev=param)


def gelu_activating(x):
    """GELU activation function"""
    return 0.5*x*(1.0+tf.math.erf(x/tf.math.sqrt(2.0)))


def hswish_activating(x):
    """Hard swish activation function"""
    return x*tf.nn.relu6(x+np.float32(3))*np.float32(1./6.)


def hsigmoid_activating(x):
    """Hard sigmoid activation function"""
    return tf.nn.relu6(x+3.)*0.16667


"""Layers"""


class Normalization(keras.layers.Layer):
    """Normalization layer"""
    def __init__(self, method, eps=1e-3, init=None, **kwargs):
        super(Normalization, self).__init__(**kwargs)
        self.method, self.epsilon, self.init, self.gamma, self.beta = method, eps, init, None, None
        self.axis = [-1] if method == 'layer' else [1, 2] if method == 'instance' else None

        if method not in ['layer', 'instance']:
            raise Exception('Unrecognized method "{}".'.format(method))

    def build(self, size):
        size1 = size[-1] if self.method == 'layer' else size[-1:] if self.method == 'instance' else None
        self.gamma = self.add_weight('gamma', size1, None, 'ones' if self.init is None else self.init)
        self.beta = self.add_weight('beta', size1, None, 'zeros')

    def call(self, x, **kwargs):
        m1, v1 = tf.nn.moments(x, self.axis, None, True)
        return self.gamma*(x-m1)*tf.math.rsqrt(v1+self.epsilon)+self.beta


class Attention(keras.layers.Layer):
    """Attention layer"""
    def __init__(self, bname, lname, head, size, attdrop=0., drop=0., eps=1e-6, ninf=-1e4, mlm=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.mlm, self.head, self.size, self.dim, self.ninf = mlm, head, size, head*size, ninf
        self.wq = keras.layers.Dense(self.dim*(1 if mlm else 3), None, True, w_initializing(), name=bname+lname[0])
        self.wk = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+lname[1]) if mlm else None
        self.wv = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+lname[2]) if mlm else None
        self.dense = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+lname[3])
        self.norm = keras.layers.LayerNormalization(-1, eps, name=bname+lname[4])
        self.attdrop = keras.layers.Dropout(attdrop)
        self.drop = keras.layers.Dropout(drop)

    def transposing(self, x):
        return tf.transpose(tf.reshape(x, [-1, tf.shape(x)[1], self.head, self.size]), [0, 2, 1, 3])

    def masking(self, mask, reg):
        m1 = mask[:, tf.newaxis, tf.newaxis, :] if not reg else mask[tf.newaxis, tf.newaxis, :, :]
        return tf.cast(m1, tf.float32)*self.ninf

    def calculating(self, q, k, v, mask, reg, training):
        a1 = tf.matmul(self.transposing(q), self.transposing(k), transpose_b=True)
        a1 = a1/tf.math.sqrt(tf.cast(self.size, tf.float32))
        a1 = tf.nn.softmax(a1+self.masking(mask, reg) if mask is not None else a1, axis=-1)
        return tf.matmul(self.attdrop(a1, training=training), self.transposing(v)), a1

    def propagating(self, x, mask=None, past=None, training=False):
        if self.mlm:
            x1, a1 = self.calculating(self.wq(x), self.wk(x), self.wv(x), mask, False, training)
            x1 = tf.reshape(tf.transpose(x1, [0, 2, 1, 3]), [-1, tf.shape(x1)[2], self.dim])
            return self.norm(x+self.drop(self.dense(x1), training=training)), a1, None
        else:
            q1, k1, v1 = tf.split(self.wq(self.norm(x)), 3, 2)
            p1 = tf.stack([k1, v1], 1)
            k2 = tf.concat([past[:, 0], k1], -2) if past is not None else k1
            v2 = tf.concat([past[:, 1], v1], -2) if past is not None else v1
            m1 = tf.range(tf.shape(q1)[1])[:, tf.newaxis] < tf.range(tf.shape(k2)[1])-tf.shape(k2)[1]+tf.shape(q1)[1]
            x1, a1 = self.calculating(q1, k2, v2, m1 if mask is None else mask, mask is None, training)
            x1 = tf.reshape(tf.transpose(x1, [0, 2, 1, 3]), [-1, tf.shape(x1)[2], self.dim])
            return x+self.drop(self.dense(x1), training=training), a1, p1


class TransEncoder(keras.layers.Layer):
    """Transformer encoder layer"""
    def __init__(self, bname, lname, head, size, dff, attdrop=0., drop=0., act='relu', eps=1e-6, mlm=True, **kwargs):
        super(TransEncoder, self).__init__(**kwargs)
        self.mlm, self.dim, self.dff = mlm, int(head*size), dff
        self.att = Attention(bname, lname, head, size, attdrop, drop, eps, mlm=mlm)
        self.norm = keras.layers.LayerNormalization(-1, eps, name=bname+lname[7])
        self.dense1 = keras.layers.Dense(dff, act, True, w_initializing(), name=bname+lname[5])
        self.dense2 = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+lname[6])
        self.drop = keras.layers.Dropout(drop)

    def propagating(self, x, mask=None, past=None, training=False):
        x1, a1, p1 = self.att.propagating(x, mask, past, training)
        x2 = self.drop(self.dense2(self.dense1(x1 if self.mlm else self.norm(x1))), training=training)
        return self.norm(x1+x2) if self.mlm else (x1+x2, p1)


class Embedding(keras.layers.Layer):
    """Embedding layer"""
    def __init__(self, bname, lname, voc, dim, maxlen=512, seg=2, drop=0., eps=1e-6, mlm=True, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.mlm, self.dim = mlm, dim
        self.emb = self.add_weight(bname+lname[0], (voc, dim), None, w_initializing())
        self.posemb = self.add_weight(bname+lname[1], (maxlen, dim), None, w_initializing())
        self.segemb = keras.layers.Embedding(seg, dim, w_initializing(), name=bname+lname[2]) if mlm else None
        self.norm = keras.layers.LayerNormalization(-1, eps, name=bname+lname[3]) if mlm else None
        self.drop = keras.layers.Dropout(drop)

    def propagating(self, x, seg=None, pos=None, training=False):
        p1 = tf.slice(self.posemb, [0, 0], [tf.shape(x)[1], -1]) if pos is None else tf.gather(self.posemb, pos)
        e1 = tf.gather(self.emb, x)+(self.segemb(seg) if self.mlm else 0.)+p1
        return self.drop(self.norm(e1) if self.mlm else e1, training=training)


class Bottleneck(keras.layers.Layer):
    """Bottleneck layer"""
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


class CRF(keras.layers.Layer):
    """CRF layer"""
    def __init__(self, dim, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.transition = self.add_weight('crf/transition', (dim, dim), None, 'glorot_uniform')

    def scanning(self, _state, _inputs):
        _state = tf.expand_dims(_state, 2)
        return _inputs+tf.reduce_logsumexp(_state+tf.expand_dims(self.transition, 0), 1)

    def normalizing(self, p, mask):
        a1 = tf.transpose(tf.scan(self.scanning, tf.transpose(p[:, 1:, :], [1, 0, 2]), p[:, 0, :]), [1, 0, 2])
        m1 = tf.argmin(tf.concat([mask, [[0]]*mask.shape[0]], 1), 1, tf.int32)-2
        return tf.reduce_logsumexp(tf.gather_nd(a1, tf.stack([tf.range(m1.shape[0]), m1], 1)), 1)

    def calculating(self, y, p, mask):
        y1 = tf.cast(y, tf.float32)*tf.expand_dims(1-tf.cast(mask, tf.float32), 2)
        s1 = tf.reduce_sum(y1*p, [1, 2])
        s2 = tf.reduce_sum(tf.matmul(y1[:, :-1], y1[:, 1:], True)*self.transition, [1, 2])
        return self.normalizing(p, 1-mask)-s1-s2

    def decoding(self, batch, mask):
        tran1 = np.expand_dims(self.transition.numpy(), 0)
        mask1 = np.argmax(np.concatenate([mask, [[1]]*mask.shape[0]], 1), 1)
        path1, stat1 = np.zeros_like(batch), np.zeros_like(batch)
        stat1[:, 0] = batch[:, 0]

        for i1 in range(1, np.max(mask1)):
            s1 = np.expand_dims(stat1[:, i1-1], 2)+tran1
            stat1[:, i1], path1[:, i1] = np.max(s1, 1)+batch[:, i1], np.argmax(s1, 1)

        rang1 = np.stack([np.arange(mask1.shape[0]), mask1-1]).T
        inde1 = np.argmax(tf.gather_nd(stat1, rang1), 1)
        return [path1[i1, :, inde1[i1]][1:rang1[i1, 1]+1].tolist()+[inde1[i1]*1.0] for i1 in range(len(inde1))]


"""Models"""


class BERT(keras.layers.Layer):
    """BERT model"""
    def __init__(self, config, model='bert', mode='seq', **kwargs):
        super(BERT, self).__init__(**kwargs)
        self.model, self.mode, self.cat = model.lower(), mode.lower(), 0
        self.param = json.load(open(config)) if type(config) is str else config
        self.embsize = self.param.get('embedding_size', self.param['hidden_size'])
        self.act = gelu_activating if self.param['hidden_act'] == 'gelu' else self.param['hidden_act']
        self.replacement = {'embeddings/embeddings': 'embeddings'}

        if self.model in ['bert', 'albert', 'electra', 'roberta'] or self.model.split('*')[0] in ['bert', 'albert']:
            self.cat = 1 if self.model.split('*')[0] == 'albert' else 0
            self.hd = 'bert' if self.model in ['albert', 'roberta'] else self.model.split('*')[-1]
            self.namep = self.hd+('/encoder/embedding_hidden_mapping_in' if self.cat else '/embeddings_project')
            self.namee = ['/word_embeddings', '/position_embeddings', '/token_type_embeddings', '/LayerNorm']
            self.namel = ['cls/predictions/'+i1 for i1 in ['transform/dense', 'transform/LayerNorm', 'output_bias']]
            self.namea = [
                '/attention/self/query',
                '/attention/self/key',
                '/attention/self/value',
                '/attention/output/dense',
                '/attention/output/LayerNorm',
                '/intermediate/dense',
                '/output/dense',
                '/output/LayerNorm']
            self.nameb = [
                '/attention_1/self/query',
                '/attention_1/self/key',
                '/attention_1/self/value',
                '/attention_1/output/dense',
                '/LayerNorm',
                '/ffn_1/intermediate/dense',
                '/ffn_1/intermediate/output/dense',
                '/LayerNorm_1']
            self.embedding = Embedding(
                self.hd+'/embeddings',
                self.namee,
                self.param['vocab_size'],
                self.embsize,
                self.param['max_position_embeddings'],
                self.param['type_vocab_size'],
                float(self.param['hidden_dropout_prob']))
            self.projection = keras.layers.Dense(
                self.param['hidden_size'],
                kernel_initializer=w_initializing(),
                name=self.namep) if self.embsize != self.param['hidden_size'] else None
            self.encoder = [TransEncoder(
                self.hd+('/encoder/transformer/group_0/inner_group_0' if self.cat else '/encoder/layer_'+str(i1)),
                self.nameb if self.cat else self.namea,
                self.param['num_attention_heads'],
                self.param['hidden_size']//self.param['num_attention_heads'],
                self.param['intermediate_size'],
                float(self.param['attention_probs_dropout_prob']),
                float(self.param['hidden_dropout_prob']),
                self.act) for i1 in range(1 if self.cat else self.param['num_hidden_layers'])]

            if self.mode == 'mlm':
                self.dense = keras.layers.Dense(self.embsize, self.act, True, w_initializing(), name=self.namel[0])
                self.norm = keras.layers.LayerNormalization(-1, 1e-6, name=self.namel[1])
                self.outbias = self.add_weight(self.namel[2], self.param['vocab_size'], None, 'zeros')

        else:
            raise Exception('Unrecognized model "{}".'.format(self.model))

    def loading(self, ckpt):
        _ = self.propagating(tf.ones((2, 2), tf.int32), tf.zeros((2, 2)), tf.zeros((2, 2)))
        r1 = re.compile('|'.join(map(re.escape, self.replacement)))
        n1 = [r1.sub((lambda x1: self.replacement[x1.group(0)]), i1.name[:-2]) for i1 in self.weights]
        keras.backend.batch_set_value(zip(self.weights, [tf.train.load_variable(ckpt, i1) for i1 in n1]))

    def propagating(self, x, seg, mask, training=False, softmax=True):
        x1, x2 = self.embedding.propagating(x, seg, training=training), None
        x1 = self.projection(x1) if self.projection else x1

        for i1 in range(self.param['num_hidden_layers']):
            x1 = self.encoder[0 if self.cat == 1 else i1].propagating(x1, mask, training=training)

        if self.mode == 'mlm':
            x2 = self.norm(self.dense(x1))
            x2 = tf.nn.bias_add(tf.matmul(x2, self.embedding.emb, transpose_b=True), self.outbias)
            x2 = tf.nn.softmax(x2) if softmax else x2

        return (x2, x1) if self.mode == 'mlm' else x1 if self.mode == 'seq' else x1[:, 0, :]


class GPT(keras.layers.Layer):
    """GPT model"""
    def __init__(self, config, **kwargs):
        super(GPT, self).__init__(**kwargs)
        self.param = json.load(open(config)) if type(config) is str else config
        self.ninf, self.eos, self.end = -1e6, 50256, [50256]
        self.replacement = {'kernel': 'w', 'bias': 'b', 'gamma': 'g', 'beta': 'b', 'lnorm': 'ln', '/embeddings': ''}
        self.norm = keras.layers.LayerNormalization(-1, 1e-6, name='model/lnorm_f')
        self.embedding = Embedding(
            'model', ['/wte', '/wpe'], self.param['n_vocab'], self.param['n_embd'], self.param['n_ctx'], mlm=False)
        self.encoder = [TransEncoder(
            'model/h'+str(i1),
            ['/attn/c_attn', '', '', '/attn/c_proj', '/lnorm_1', '/mlp/c_fc', '/mlp/c_proj', '/lnorm_2'],
            self.param['n_head'],
            self.param['n_embd']//self.param['n_head'],
            self.param['n_embd']*4,
            act=self.param.get('activation', gelu_activating),
            mlm=False) for i1 in range(self.param['n_layer'])]

    def loading(self, ckpt):
        _ = self.propagating(tf.ones((2, 2), tf.int32))
        r1 = re.compile('|'.join(map(re.escape, self.replacement)))
        n1 = [r1.sub((lambda x1: self.replacement[x1.group(0)]), i1.name[:-2]) for i1 in self.weights]
        keras.backend.batch_set_value(zip(self.weights, [tf.squeeze(tf.train.load_variable(ckpt, i1)) for i1 in n1]))

    def propagating(self, x, mask=None, pos=None, past=None, training=False, softmax=True):
        p1, p2 = (tf.unstack(past, axis=1) if past is not None else [None]*self.param['n_layer']), []
        t1 = pos if pos is not None else tf.repeat([past.shape[3]], x.shape[0], 0) if past is not None else None
        x1 = self.embedding.propagating(x, None, t1, training)

        for i1 in range(self.param['n_layer']):
            x1, a1 = self.encoder[i1].propagating(x1, mask, p1[i1], training)
            p2.append(a1)

        x1, h1 = self.norm(x1), tf.stack(p2, 1)
        x2 = tf.matmul(x1, self.embedding.emb, transpose_b=True)
        return tf.nn.softmax(x2) if softmax else x2, x1, h1

    def sampling(self, score, cur, beam, k, p, first):
        s1 = cur+self.ninf*(1.-tf.reduce_sum(tf.one_hot(tf.math.top_k(cur, k)[1], self.param['n_vocab']), 1))
        i1 = tf.argsort(s1, -1, 'DESCENDING')
        f1 = tf.math.cumsum(tf.nn.softmax(tf.gather(s1, i1, axis=1, batch_dims=1)), -1) > p
        f1 = tf.gather(tf.concat([tf.zeros_like(f1[:, :1]), f1[:, :-1]], -1), tf.argsort(i1), axis=1, batch_dims=1)
        s2 = s1+tf.cast(f1, tf.float32)*self.ninf
        t1 = tf.reshape(tf.random.categorical(s2, beam if first else 1, tf.int32), [-1, beam, 1])
        p1 = tf.reshape(tf.repeat(tf.range(score.shape[0]), beam if first else 1), [-1, beam, 1])
        i2 = tf.argsort(tf.gather_nd(score, tf.concat([p1, t1], -1)), -1, 'DESCENDING')
        t1, p1 = tf.gather(t1, i2, axis=1, batch_dims=1), tf.gather(p1, i2, axis=1, batch_dims=1)
        return tf.reshape(t1, [-1, 1]), tf.reshape(p1, [-1, 1])

    def searching(self, score, beam, flag):
        v1, i1 = tf.math.top_k(tf.reshape(score, [-1, (beam**flag)*self.param['n_vocab']]), beam)
        p1 = tf.reshape(tf.expand_dims(tf.range(v1.shape[0])*(beam**flag), 1)+(i1//self.param['n_vocab']), [-1, 1])
        return tf.reshape(i1 % self.param['n_vocab'], [-1, 1]), p1

    def calculating(self, score, cur, beam, k, p, length, penalty, first):
        s1 = score/length**penalty
        t1, p1 = self.sampling(s1, cur, beam, k, p, first) if k > 1 else self.searching(s1, beam, not first)
        i1 = tf.expand_dims(tf.concat([p1, t1], -1), 1)
        return t1, p1[:, 0], tf.gather_nd(score, i1), tf.gather_nd(length, i1)

    def updating(self, step, pred, mapping, record, beam, final):
        b1 = tf.cast([[1] if i1[0] in self.end else [0] for i1 in step.numpy()], tf.float32)
        e1 = np.where(np.all(np.reshape(b1, [-1, beam]), -1) > (-1 if final else 0))[0]
        e2 = (np.reshape(np.delete(np.arange(b1.shape[0]//beam), e1), [-1, 1])*beam+np.arange(beam)).flatten()

        for i1 in e1:
            record[mapping[i1]] = pred.numpy()[i1*beam:(i1+1)*beam].tolist()

        m1 = np.delete(mapping, e1)
        return e1, e2, b1, m1, record

    def generating(self, x, mask, pos, beam=5, k=1, p=0.9, temp=1.0, penalty=1.0, maxlen=10, best=False):
        x1, b1, m1, p1 = x, x.shape[0], tf.repeat(mask, beam, 0), tf.repeat(pos, beam, 0)
        scor1, leng1, past1, fini1, pred1 = 0., tf.cast(pos, tf.float32)+1., None, None, None
        list1, list2, i1 = np.arange(b1), [None]*b1, 0
        mask1 = tf.repeat(tf.one_hot([self.eos], self.param['n_vocab'], 0., self.ninf), b1*beam, 0)
        appe1 = tf.one_hot([self.eos], self.param['n_vocab'], 0., 1.)

        while i1 < maxlen and len(list1) > 0:
            x2, _, h1 = self.propagating(x1, m1 if i1 > 0 else None, p1+i1 if i1 else None, past1, False, False)
            s1 = tf.nn.log_softmax(tf.squeeze(x2/temp, 1)+fini1*mask1 if i1 else tf.gather_nd(x2/temp, pos, 1))
            x1, h2, scor1, leng1 = self.calculating(scor1+s1, s1, beam, k, p, leng1+appe1, penalty, i1 == 0)
            m1 = tf.concat([m1, [[0]]*x1.shape[0]], -1)
            past1 = tf.gather(tf.concat([past1, h1], -2) if i1 else h1, h2)
            pred1, i1 = tf.concat([tf.gather(pred1, h2), x1], -1) if i1 else x1, i1+1
            e1, e2, fini1, list1, list2 = self.updating(x1, pred1, list1, list2, beam, i1 == maxlen)

            if len(e1) > 0 and i1 < maxlen:
                x1, m1, p1, past1, scor1, leng1, mask1, fini1, pred1 = [
                    tf.gather(j1, e2) for j1 in [x1, m1, p1, past1, scor1, leng1, mask1, fini1, pred1]]

        return [[i1[0]] for i1 in list2] if best else list2


class MobileNet(keras.layers.Layer):
    """MobileNet V3 model"""
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
            caxis,
            self.param[i1][4],
            self.param[i1][5]) for i1 in range(1, len(self.param))]

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
        x1 = self.act(self.conv3(tf.reshape(self.pool(x1), [-1, 1, 1, x1.shape[-1]])))
        return tf.nn.softmax(self.conv4(x1)) if softmax else self.conv4(x1)


"""Optimizers"""


class AdamW(keras.optimizers.Optimizer):
    """AdamW optimizer"""
    def __init__(self, step, lrate=1e-3, b1=0.9, b2=0.999, drate=1e-2, lmode=0, ldecay=None, name='AdamW', **kwargs):
        super(AdamW, self).__init__(name, **kwargs)
        self.step, self.drate, self.lmode, self.ldecay, self.epsilon = step, drate, lmode, ldecay, 1e-6
        self.spec = ['bias', 'normalization', 'lnorm', 'layernorm']
        self._set_hyper('learning_rate', lrate)
        self._set_hyper('beta_1', b1)
        self._set_hyper('beta_2', b2)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')

    @staticmethod
    def _rate_sch(rate, step, total):
        warm1 = total*0.1
        return tf.where(step < warm1, rate*step/warm1, rate*(total-step)/(total-warm1))

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamW, self)._prepare_local(var_device, var_dtype, apply_state)
        rate1 = apply_state[(var_device, var_dtype)]['lr_t']
        beta1 = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta2 = tf.identity(self._get_hyper('beta_2', var_dtype))
        apply_state[(var_device, var_dtype)].update(dict(
            lr=self._rate_sch(rate1, tf.cast(self.iterations+1, var_dtype), self.step+1),
            epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
            beta_1=beta1,
            beta_1_minus=1-beta1,
            beta_2=beta2,
            beta_2_minus=1-beta2))

    def _resource_apply_base(self, grad, var, indices=None, apply_state=None):
        devi1, type1, name1 = var.device, var.dtype.base_dtype, var.name
        spec1 = any(c1 in name1.lower() for c1 in self.spec)
        coef1 = ((apply_state or {}).get((devi1, type1)) or self._fallback_apply_state(devi1, type1))
        m1, v1, r1 = self.get_slot(var, 'm'), self.get_slot(var, 'v'), 1.0

        if indices is None:
            m2 = m1.assign(coef1['beta_1']*m1+coef1['beta_1_minus']*grad, self._use_locking)
            v2 = v1.assign(coef1['beta_2']*v1+coef1['beta_2_minus']*grad*grad, self._use_locking)

        else:
            m2 = m1.assign(coef1['beta_1']*m1, self._use_locking)
            v2 = v1.assign(coef1['beta_2']*v1, self._use_locking)

            with tf.control_dependencies([m2, v2]):
                m2 = self._resource_scatter_add(m1, indices, coef1['beta_1_minus']*grad)
                v2 = self._resource_scatter_add(v1, indices, coef1['beta_2_minus']*grad*grad)

        u1 = m2/(tf.sqrt(v2)+coef1['epsilon'])
        u1 = u1 if spec1 else u1+self.drate*var

        if self.lmode == 1 and not spec1:
            n1, n2 = tf.norm(var, 2), tf.norm(u1, 2)
            r1 = tf.where(tf.greater(n1, 0.), tf.where(tf.greater(n2, 0.), n1/n2, 1.), 1.)

        if self.lmode == 2 and not spec1:
            r1 = self.ldecay.get(([c2 for c2 in self.ldecay.keys() if c2 in name1]+[''])[0], r1)

        return tf.group(*[var.assign_sub(r1*coef1['lr']*u1, self._use_locking), m2, v2])

    def _resource_apply_dense(self, grad, var, apply_state=None, **kwargs):
        return self._resource_apply_base(grad, var, None, apply_state)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None, **kwargs):
        return self._resource_apply_base(grad, var, indices, apply_state)

    def get_config(self):
        conf1 = super(AdamW, self).get_config()
        conf1.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'decaying_rate': self.drate,
            'epsilon': self.epsilon,
            'step': self.step})
        return conf1


class AdamWV2(keras.optimizers.Adam):
    """AdamW optimizer V2"""
    def __init__(self, step, lrate=1e-3, drate=1e-2, name='AdamW', **kwargs):
        super(AdamWV2, self).__init__(learning_rate=lrate, name=name, **kwargs)
        self.step, self.drate, self.spec = step, drate, ['bias', 'normalization', 'lnorm', 'layernorm']

    @staticmethod
    def _rate_sch(rate, step, total):
        warm1 = total*0.1
        return tf.where(step < warm1, rate*step/warm1, rate*(total-step)/(total-warm1))

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamWV2, self)._prepare_local(var_device, var_dtype, apply_state)
        rate1 = self._rate_sch(1., tf.cast(self.iterations+1, var_dtype), self.step+1)
        apply_state[(var_device, var_dtype)]['lr_t'] *= rate1
        apply_state[(var_device, var_dtype)]['lr'] *= rate1

    def _resource_apply_base(self, var, apply_state=None):
        devi1, type1, spec1 = var.device, var.dtype.base_dtype, any(c1 in var.name.lower() for c1 in self.spec)
        coef1 = ((apply_state or {}).get((devi1, type1)) or self._fallback_apply_state(devi1, type1))
        return tf.no_op if spec1 else var.assign_sub(coef1['lr_t']*var*self.drate, use_locking=self._use_locking)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        deca1 = self._resource_apply_base(var, apply_state)

        with tf.control_dependencies([deca1]):
            return super(AdamWV2, self)._resource_apply_dense(grad, var, apply_state)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        deca1 = self._resource_apply_base(var, apply_state)

        with tf.control_dependencies([deca1]):
            return super(AdamWV2, self)._resource_apply_sparse(grad, var, indices, apply_state)

    def get_config(self):
        conf1 = super(AdamWV2, self).get_config()
        conf1.update({'decaying_rate': self.drate, 'step': self.step})
        return conf1


class MovingAverage:
    """Exponential moving average strategy"""
    def __init__(self, var, decay=0.99):
        self.decay, self.var = decay, [keras.backend.zeros(w.shape) for w in var]
        keras.backend.batch_set_value(zip(self.var, keras.backend.batch_get_value(var)))
        self.updating(var)

    def updating(self, var):
        for w1, w2 in zip(self.var, var):
            keras.backend.moving_average_update(w1, w2, self.decay)

    def setting(self, var):
        keras.backend.batch_set_value(zip(var, keras.backend.batch_get_value(self.var)))


"""Tools"""


class Tokenizer:
    """Simple text tokenizer"""
    def __init__(self, mlm=True, lower=True, num=True):
        self.mlm, self.lower, self.num, self.vocab, self.w, self.e, self.d = mlm, lower, num, None, None, None, None
        self.bos, self.sep, self.pad, self.unk = ['[CLS]', '[SEP]', '[PAD]', '[UNK]'] if mlm else ['<|endoftext|>']*4
        self.pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.ch = [[33, 47], [58, 64], [91, 96], [123, 126], [0x4E00, 0x9FFF], [0x3400, 0x4DBF], [0x20000, 0x2A6DF],
                   [0x2A700, 0x2B73F], [0x2B740, 0x2B81F], [0x2B820, 0x2CEAF], [0xF900, 0xFAFF], [0x2F800, 0x2FA1F]]

    def loading(self, path, enc='utf-8'):
        b1 = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
        o1 = [i1 for i1 in range(2**8) if i1 not in b1]
        c1 = [chr(i1) for i1 in b1[:]+[2**8+j1 for j1 in range(len(o1))]]
        self.vocab = dict([(j1.strip(), i1) for i1, j1 in enumerate(open(path, encoding=enc))]) if self.mlm else {}
        self.vocab = self.vocab if self.mlm else json.load(open(path, encoding=enc))
        self.w, self.e, self.d = list(self.vocab.keys()), dict(zip(b1+o1, c1)), dict(zip(c1, b1+o1))

    def splitting(self, token):
        toke1, toke2 = [], (re.findall(r'Ġ\d|\d|\D+', token) if self.num else [token])

        for i1, char1 in enumerate(toke2):
            star1, endi1 = 0, 0

            while star1 < len(char1):
                for endi1 in range(len(char1), star1, -1):
                    subt1 = ('##' if self.mlm and (i1 > 0 or star1 > 0) else '')+char1[star1:endi1]

                    if subt1 in self.vocab:
                        toke1, star1 = toke1+[subt1], endi1
                        break

                if star1 != endi1:
                    return [self.unk]

        return toke1

    def separating(self, text, pre):
        if self.mlm:
            func1 = (lambda x1: any(i1[0] <= ord(x1) <= i1[1] for i1 in self.ch))
            orig1 = [' '+c1+' ' if unicodedata.category(c1).startswith('P') or func1(c1) else c1 for c1 in text]
            text1 = re.sub(r'\[ mask ]|\[ MASK ]', '[MASK]', ''.join(orig1)).strip().split()
        else:
            orig1 = regex.findall(self.pat, (' ' if pre and text[0] != ' ' else '')+text)
            text1 = [''.join([self.e[j1] for j1 in i1.encode('utf-8')]) for i1 in orig1]

        text2 = sum([self.splitting(t1) for t1 in text1], [])
        return text2, len(text2)

    def processing(self, a, b, maxlen, pre):
        a1, l1 = self.separating(a.lower() if self.lower else a, pre)
        b1, l2 = self.separating(b.lower() if self.lower else b, pre) if b else ([], 0)
        a2 = a1[:min(l1, int(np.ceil(maxlen/2)))] if l1 < l2 else a1[:max(maxlen-l2, int(np.ceil(maxlen/2)))]
        return a2, b1[:min(l2, maxlen//2)] if l2 < l1 else b1[:max(maxlen-l1, maxlen//2)]

    def encoding(self, a, b=None, maxlen=64, bos=True, sep=True, pad=True, pre=True, length=False):
        a1, b1 = self.processing(a, b, maxlen-bos-sep-(b is not None and sep), pre)
        a1, b1 = [self.bos]*bos+a1+[self.sep]*sep, b1+[self.sep]*(b is not None and sep)
        l1, l2 = len(a1), len(b1)
        padd1 = maxlen-l1-l2 if pad else 0
        sent1 = [self.vocab.get(i1, self.vocab[self.unk]) for i1 in (a1+b1+[self.pad]*padd1)]
        segm1 = [0]*l1+[1]*l2+[0 if not b else 1]*padd1
        mask1 = [0]*(l1+l2)+[1]*padd1
        return (sent1, segm1, mask1, l1 if b is None else (l1, l2)) if length else (sent1, segm1, mask1)

    def decoding(self, token):
        toke1 = [j1 for j1 in [self.w[i1] for i1 in token] if j1 not in [self.bos, self.sep, self.pad]]
        func1 = (lambda x1: re.sub(r' ##', '', x1))
        func2 = (lambda x1: bytearray([self.d[i1] for i1 in x1]).decode('utf-8', errors='replace'))
        return func1(' '.join(toke1)) if self.mlm else func2(''.join(toke1))


"""Tests"""


def bert_testing(sentence=None, mlm=False):
    """Test of BERT"""
    toke1 = Tokenizer()
    toke1.loading('./models/bert_base_ch/vocab.txt')
    bert1 = BERT('./models/bert_base_ch/bert_config.json', 'bert', 'mlm' if mlm else 'seq')
    bert1.loading('./models/bert_base_ch/bert_model.ckpt')
    text1, segm1, mask1 = toke1.encoding('感觉[MASK]就像一只猪。' if not sentence else sentence, pad=False)
    pred1 = bert1.propagating(np.array([text1]), np.array([segm1]), np.array([mask1]))
    print(''.join([toke1.w[i1] for i1 in list(np.array(np.argmax(pred1[0], axis=-1)[0]))[1:-1]]) if mlm else pred1)


def gpt_testing(sent=None, lm=False):
    """Test of GPT"""
    toke1 = Tokenizer(False, False)
    toke1.loading('./models/gpt_base_en/encoder.json')
    g1 = GPT('./models/gpt_base_en/hparams.json')
    g1.loading('./models/gpt_base_en/model.ckpt')
    t1, s1, m1, l1 = toke1.encoding(sent if sent else 'Have a good day.', None, 64, False, False, False, True, True)
    print(g1.generating(np.array([t1]), [m1], [[l1-1]]) if lm else g1.propagating(np.array([t1]))[1])


def mobile_testing(image=None):
    """Test of MobileNet"""
    imag1 = tf.image.decode_jpeg(tf.io.read_file('./models/v3_small_float/panda.jpg' if not image else image), 3)
    imag1 = tf.expand_dims(tf.image.resize(tf.cast(imag1, tf.float32)/128.-1., [224, 224]), 0)
    mnet1 = MobileNet(1, -1, 1001)
    mnet1.loading('./models/v3_small_float/model-388500')
    inpu1 = keras.Input(shape=(224, 224, 3))
    mode1 = keras.Model(inputs=inpu1, outputs=mnet1.propagating(inpu1))
    print(np.flipud(np.argsort(mode1.predict(imag1)[0, 0, 0, :]))[:5])
