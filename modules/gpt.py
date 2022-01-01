# -*- coding: utf-8 -*-

"""
GPT-2 (https://github.com/openai/gpt-2)

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


def gelunew_activating(x):
    return 0.5*x*(1.0+tf.tanh((np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3)))))


class Attention(keras.layers.Layer):
    def __init__(self, bname, lname, head, size, attdrop=0., drop=0., eps=1e-6, ninf=-1e4, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.head, self.size, self.dim, self.ninf = head, size, head*size, ninf
        self.wq = keras.layers.Dense(self.dim*3, None, True, w_initializing(), name=bname+lname[0])
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

    def propagating(self, x, mask=None, past=None, hist=None, training=False):
        q1, k1, v1 = tf.split(self.wq(self.norm(x)), 3, 2)
        k1 = tf.concat([hist[0], k1], -2) if hist is not None else k1
        v1 = tf.concat([hist[1], v1], -2) if hist is not None else v1
        k2 = tf.concat([past[:, 0], k1], -2) if past is not None else k1
        v2 = tf.concat([past[:, 1], v1], -2) if past is not None else v1
        m1 = tf.range(tf.shape(q1)[1])[:, tf.newaxis] < tf.range(tf.shape(k2)[1])-tf.shape(k2)[1]+tf.shape(q1)[1]
        x1, a1 = self.calculating(q1, k2, v2, m1 if mask is None else mask, mask is None, training)
        x1 = tf.reshape(tf.transpose(x1, [0, 2, 1, 3]), [-1, tf.shape(x1)[2], self.dim])
        return x+self.drop(self.dense(x1), training=training), a1, tf.stack([k1, v1], 1)


class TransEncoder(keras.layers.Layer):
    def __init__(self, bname, lname, head, size, dff, attdrop=0., drop=0., act='relu', eps=1e-6, **kwargs):
        super(TransEncoder, self).__init__(**kwargs)
        self.att = Attention(bname, lname, head, size, attdrop, drop, eps)
        self.norm = keras.layers.LayerNormalization(-1, eps, name=bname+lname[7])
        self.dense1 = keras.layers.Dense(dff, act, True, w_initializing(), name=bname+lname[5])
        self.dense2 = keras.layers.Dense(int(head*size), None, True, w_initializing(), name=bname+lname[6])
        self.drop = keras.layers.Dropout(drop)

    def propagating(self, x, mask=None, past=None, hist=None, training=False):
        x1, a1, p1 = self.att.propagating(x, mask, past, hist, training)
        return x1+self.drop(self.dense2(self.dense1(self.norm(x1))), training=training), p1


class Embedding(keras.layers.Layer):
    def __init__(self, lname, voc, dim, maxlen=512, drop=0., **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.emb = self.add_weight(lname[0], (voc, dim), None, w_initializing())
        self.posemb = self.add_weight(lname[1], (maxlen, dim), None, w_initializing())
        self.drop = keras.layers.Dropout(drop)

    def propagating(self, x, pos=None, training=False):
        p1 = tf.slice(self.posemb, [0, 0], [tf.shape(x)[1], -1]) if pos is None else tf.gather(self.posemb, pos)
        return self.drop(tf.gather(self.emb, x)+p1, training=training)


class GPT(keras.layers.Layer):
    def __init__(self, config, head='', **kwargs):
        super(GPT, self).__init__(**kwargs)
        self.ninf, self.eos, self.end = -1e6, 50256, [50256]
        self.param = json.load(open(config)) if type(config) is str else config
        self.rpl = {'/': '.', 'kernel': 'weight', 'gamma': 'weight', 'beta': 'bias'}
        self.act = {'gelu': gelu_activating, 'gelu_new': gelunew_activating}
        self.norm = keras.layers.LayerNormalization(-1, self.param['layer_norm_epsilon'], name=head+'ln_f')
        self.embedding = Embedding(
            [head+'wte.weight', head+'wpe.weight'],
            self.param['vocab_size'],
            self.param['n_embd'],
            self.param['n_ctx'],
            self.param['embd_pdrop'])
        self.encoder = [TransEncoder(
            head+'h/'+str(i1),
            ['/attn/c_attn', '', '', '/attn/c_proj', '/ln_1', '/mlp/c_fc', '/mlp/c_proj', '/ln_2'],
            self.param['n_head'],
            self.param['n_embd']//self.param['n_head'],
            self.param['n_embd']*4,
            self.param['attn_pdrop'],
            self.param['resid_pdrop'],
            self.act.get(self.param['activation_function'], self.param['activation_function']),
            self.param['layer_norm_epsilon']) for i1 in range(self.param['n_layer'])]

    def loading(self, pth):
        _ = self.propagating(tf.ones((2, 2), tf.int32))
        r1, l1 = re.compile('|'.join(map(re.escape, self.rpl))), torch.load(pth)
        n1 = [r1.sub((lambda x1: self.rpl[x1.group(0)]), i1.name[:-2]) for i1 in self.weights]
        t1 = [l1[i1].numpy() for i1 in n1]
        keras.backend.batch_set_value(zip(self.weights, t1))

    def propagating(self, x, mask=None, pos=None, past=None, hist=None, training=False, softmax=True):
        p1, p2 = (tf.unstack(past, axis=1) if past is not None else [None]*self.param['n_layer']), []
        t1 = pos if pos is not None else tf.repeat([past.shape[3]], x.shape[0], 0) if past is not None else None
        x1 = self.embedding.propagating(x, t1, training)

        for i1 in range(self.param['n_layer']):
            x1, a1 = self.encoder[i1].propagating(x1, mask, p1[i1], None if hist is None else hist[i1], training)
            p2.append(a1)

        x1, h1 = self.norm(x1), tf.stack(p2, 1)
        x2 = tf.matmul(x1, self.embedding.emb, transpose_b=True)
        return tf.nn.softmax(x2) if softmax else x2, x1, h1

    def sampling(self, score, cur, beam, k, p, first):
        s1 = cur+self.ninf*(1.-tf.reduce_sum(tf.one_hot(tf.math.top_k(cur, k)[1], self.param['vocab_size']), 1))
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
        v1, i1 = tf.math.top_k(tf.reshape(score, [-1, (beam**flag)*self.param['vocab_size']]), beam)
        p1 = tf.expand_dims(tf.range(tf.shape(v1)[0])*(beam**flag), 1)+(i1//self.param['vocab_size'])
        return tf.reshape(i1 % self.param['vocab_size'], [-1, 1]), tf.reshape(p1, [-1, 1])

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

    def generating(self, x, mask, pos, hist=None, beam=5, k=1, p=0.9, temp=1.0, penalty=1.0, maxlen=10, best=False):
        x1, b1, m1, p1 = x, x.shape[0], tf.repeat(mask, beam, 0), tf.repeat(pos, beam, 0)
        scor1, leng1, past1, fini1, pred1 = 0., tf.cast(pos, tf.float32)+1., None, None, None
        list1, list2, i1 = np.arange(b1), [None]*b1, 0
        mask1 = tf.repeat(tf.one_hot([self.eos], self.param['vocab_size'], 0., self.ninf), b1*beam, 0)
        appe1 = tf.one_hot([self.eos], self.param['vocab_size'], 0., 1.)

        while i1 < maxlen and len(list1) > 0:
            hist1 = hist if i1 == 0 and hist is not None else None
            x2, _, h1 = self.propagating(x1, m1 if i1 > 0 else None, p1+i1 if i1 else None, past1, hist1, False)
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
