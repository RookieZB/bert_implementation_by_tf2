# -*- coding: utf-8 -*-

"""
Title: BERT by TF2
Version: 0.1.1
Author: Chris Chen
Date: 2020/02/02
Update: 2020/02/28

"""

import os
import warnings
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


"""Calculation functions"""


def w_initializing(param=0.2):
    """The truncated normal initializing"""
    return keras.initializers.TruncatedNormal(stddev=param)


def gelu_activating(x):
    """The GELU activation function"""
    g1 = 0.5*(1.0+tf.tanh((np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3)))))
    return x*g1


"""Network layers"""


class Attention(keras.layers.Layer):
    """The attention layer"""
    def __init__(self, bname, lname, head, size, drop, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.head, self.size, self.ninf = head, size, -1e4
        self.wq = keras.layers.Dense(self.head*self.size, None, True, w_initializing(), name=bname+lname[0])
        self.wk = keras.layers.Dense(self.head*self.size, None, True, w_initializing(), name=bname+lname[1])
        self.wv = keras.layers.Dense(self.head*self.size, None, True, w_initializing(), name=bname+lname[2])
        self.dense = keras.layers.Dense(self.head*self.size, None, True, w_initializing(), name=bname+lname[3])
        self.norm = keras.layers.LayerNormalization(-1, 1e-6, name=bname+lname[4])
        self.drop = keras.layers.Dropout(drop)

    def transposing(self, x):
        x1 = tf.reshape(x, [-1, x.shape[1], self.head, self.size])
        return tf.transpose(x1, [0, 2, 1, 3])

    def masking(self, mask):
        l1 = mask.shape
        m1 = tf.cast(tf.reshape(mask, [-1, 1, l1[1]]), tf.float32)
        m2 = tf.ones((l1[0], l1[1], 1), tf.float32)
        return tf.expand_dims(m2*m1, axis=[1])*self.ninf

    def calculating(self, q, k, v, mask):
        atte1 = tf.matmul(q, k, transpose_b=True)/tf.math.sqrt(tf.cast(self.size, tf.float32))
        atte1 = atte1+self.masking(mask) if mask is not None else atte1
        atte1 = tf.nn.softmax(atte1, axis=-1)
        return tf.matmul(atte1, v), atte1

    def propagating(self, x, training, mask):
        q1, k1, v1 = self.transposing(self.wq(x)), self.transposing(self.wk(x)), self.transposing(self.wv(x))
        x1, a1 = self.calculating(q1, k1, v1, mask)
        x1 = tf.transpose(x1, [0, 2, 1, 3])
        x1 = tf.reshape(x1, [-1, x1.shape[1], self.head*self.size])
        return self.norm(x+self.drop(self.dense(x1), training=training)), a1


class Transformer(keras.layers.Layer):
    """The transformer layer"""
    def __init__(self, bname, lname, head, size, dff, attdrop=0.1, drop=0.1, act='relu', **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.dim, self.dff = int(head*size), dff
        self.att = Attention(bname, lname, head, size, attdrop)
        self.norm = keras.layers.LayerNormalization(-1, 1e-6, name=bname+lname[7])
        self.drop = keras.layers.Dropout(drop)
        self.dense1 = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+lname[6])
        self.dense2 = keras.layers.Dense(self.dff, act, True, w_initializing(), name=bname+lname[5])

    def propagating(self, x, training, mask):
        x1, a1 = self.att.propagating(x, training, mask)
        x2 = self.dense1(self.dense2(x1))
        return self.norm(x1+self.drop(x2, training=training))


class Embedding(keras.layers.Layer):
    """The embedding layer"""
    def __init__(self, bname, lname, voc, seg, dim, maxlen, drop=0.1, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.dim, self.maxlen = dim, maxlen
        self.embedding = keras.layers.Embedding(voc, dim, w_initializing(), name=bname+lname[0])
        self.segemb = keras.layers.Embedding(seg, dim, w_initializing(), name=bname+lname[1])
        self.posemb = self.add_weight(bname+lname[2], (self.maxlen, self.dim), None, w_initializing())
        self.norm = keras.layers.LayerNormalization(-1, 1e-6, name=bname+lname[3])
        self.drop = keras.layers.Dropout(drop)

    def posencoding(self, pos):
        angl0 = 1/np.power(10000, (2*(np.arange(self.dim)[np.newaxis, :]//2))/np.float32(self.dim))
        angl1 = np.arange(pos)[:, np.newaxis]*angl0
        angl1[:, 0::2] = np.sin(angl1[:, 0::2])
        angl1[:, 1::2] = np.cos(angl1[:, 1::2])
        return tf.cast(angl1[np.newaxis, ...], dtype=tf.float32)

    def propagating(self, x, seg, training):
        x1 = self.embedding(x)  # x1 = x1*tf.math.sqrt(tf.cast(self.dim, tf.float32))
        l1 = tf.shape(x)
        x2 = self.segemb(tf.zeros((l1[0], l1[1]))) if seg is None else self.segemb(seg)
        p1 = tf.slice(self.posemb, [0, 0], [l1[1], -1])
        return self.drop(self.norm(x1+x2+p1), training=training)


"""BERT models"""


class BERT(keras.layers.Layer):
    """The BERT model"""
    def __init__(self, config, albert=False, **kwargs):
        super(BERT, self).__init__(**kwargs)
        self.albert = albert
        self.param = json.load(open(config)) if type(config) is str else config
        self.act = gelu_activating if self.param['hidden_act'] == 'gelu' else self.param['hidden_act']
        self.replacement = {
            'bert/embeddings/word_embeddings/embeddings': 'bert/embeddings/word_embeddings',
            'bert/embeddings/token_type_embeddings/embeddings': 'bert/embeddings/token_type_embeddings'}
        self.namea = [
            '/attention_1/self/query',
            '/attention_1/self/key',
            '/attention_1/self/value',
            '/attention_1/output/dense',
            '/LayerNorm',
            '/ffn_1/intermediate/dense',
            '/ffn_1/intermediate/output/dense',
            '/LayerNorm_1']
        self.nameb = [
            '/attention/self/query',
            '/attention/self/key',
            '/attention/self/value',
            '/attention/output/dense',
            '/attention/output/LayerNorm',
            '/intermediate/dense',
            '/output/dense',
            '/output/LayerNorm']
        self.namee = [
            '/word_embeddings',
            '/token_type_embeddings',
            '/position_embeddings',
            '/LayerNorm']
        self.embedding = Embedding(
            'bert/embeddings',
            self.namee,
            self.param['vocab_size'],
            self.param['type_vocab_size'],
            self.param['embedding_size'] if self.albert else self.param['hidden_size'],
            self.param['max_position_embeddings'],
            float(self.param['hidden_dropout_prob']))

        if self.albert:
            self.projection = keras.layers.Dense(
                self.param['hidden_size'],
                kernel_initializer=w_initializing(),
                name='bert/encoder/embedding_hidden_mapping_in')
            self.encoder = [Transformer(
                'bert/encoder/transformer/group_0/inner_group_0',
                self.namea,
                self.param['num_attention_heads'],
                int(self.param['hidden_size']/self.param['num_attention_heads']),
                self.param['intermediate_size'],
                float(self.param['attention_probs_dropout_prob']),
                float(self.param['hidden_dropout_prob']),
                self.act)]
        else:
            self.projection = None
            self.encoder = [Transformer(
                'bert/encoder/layer_'+str(i1),
                self.nameb,
                self.param['num_attention_heads'],
                int(self.param['hidden_size']/self.param['num_attention_heads']),
                self.param['intermediate_size'],
                float(self.param['attention_probs_dropout_prob']),
                float(self.param['hidden_dropout_prob']),
                self.act) for i1 in range(self.param['num_hidden_layers'])]

    def loading(self, ckpt):
        _ = self.propagating(tf.ones((2, 2)), None, tf.zeros((2, 2)), False, False)
        tens1, name1 = self.weights, [i1.name[:-2] for i1 in self.weights]
        name1 = [i1 if i1 not in self.replacement.keys() else self.replacement[i1] for i1 in name1]
        valu1 = [tf.train.load_variable(ckpt, i1) for i1 in name1]
        keras.backend.batch_set_value(zip(tens1, valu1))

    def propagating(self, x, seg=None, mask=None, cls=False, training=False):
        x1 = self.embedding.propagating(x, seg, training)
        x1 = self.projection(x1) if self.albert else x1

        if self.albert:
            for i1 in range(self.param['num_hidden_layers']):
                x1 = self.encoder[0].propagating(x1, training, mask)

        else:
            for i1 in range(self.param['num_hidden_layers']):
                x1 = self.encoder[i1].propagating(x1, training, mask)

        return tf.reshape(x1[:, 0, :], [-1, self.param['hidden_size']]) if cls else x1


"""Training optimizers"""


class AdamW(keras.optimizers.Optimizer):
    """The AdamW optimizer"""
    def __init__(self, step, lrate=1e-3, drate=1e-2, b1=0.9, b2=0.999, eps=1e-6, lnorm=False, name='AdamW', **kwargs):
        super(AdamW, self).__init__(name, **kwargs)
        self.lnorm, self.step, self.drate, self.epsilon = lnorm, step, drate, eps or keras.backend_config.epislon()
        self._set_hyper('learning_rate', lrate)
        self._set_hyper('beta_1', b1)
        self._set_hyper('beta_2', b2)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')

    def _decayed_lr(self, var_dtype):
        r1 = self._get_hyper('learning_rate', var_dtype)
        s1 = tf.cast(self.iterations, var_dtype)
        warm1 = tf.cast(int(self.step*0.1), var_dtype)
        return tf.where(s1 < warm1, r1*s1/warm1, r1*(self.step-s1)/(self.step-warm1))

    def _resource_apply_base(self, grad, var, indices=None):
        t1, name1 = var.dtype.base_dtype, var.name
        r1 = self._decayed_lr(t1)
        e1 = tf.convert_to_tensor(self.epsilon, t1)
        m1, v1 = self.get_slot(var, 'm'), self.get_slot(var, 'v')
        beta1, beta2 = tf.identity(self._get_hyper('beta_1', t1)), tf.identity(self._get_hyper('beta_2', t1))
        # step1 = tf.cast(self.iterations+1, t1)
        # powe1, powe2 = tf.pow(beta1, step1), tf.pow(beta2, step1)
        # rate1 = rate1*tf.sqrt(1-powe2)/(1-powe1)

        if indices is None:
            m2 = m1.assign(beta1*m1+(1.0-beta1)*grad, self._use_locking)
            v2 = v1.assign(beta2*v1+(1.0-beta2)*grad*grad, self._use_locking)
        else:
            m2 = m1.assign(beta1*m1, self._use_locking)
            v2 = v1.assign(beta2*v1, self._use_locking)

            with tf.control_dependencies([m2, v2]):
                m2 = self._resource_scatter_add(m1, indices, (1.0-beta1)*grad)
                v2 = self._resource_scatter_add(v1, indices, (1.0-beta2)*grad*grad)

        u1 = m2/(tf.sqrt(v2)+e1) if 'LayerNorm' in name1 or 'bias' in name1 else m2/(tf.sqrt(v2)+e1)+self.drate*var

        if self.lnorm and 'LayerNorm' not in name1 and 'bias' not in name1:
            n1, n2 = tf.norm(var, 2), tf.norm(u1, 2)
            rati1 = tf.where(tf.greater(n1, 0.), tf.where(tf.greater(n2, 0.), n1/n2, 1.), 1.)
            return tf.group(*[var.assign_sub(rati1*r1*u1, self._use_locking), m2, v2])
        else:
            return tf.group(*[var.assign_sub(r1*u1, self._use_locking), m2, v2])

    def _resource_apply_dense(self, grad, var, **kwargs):
        return self._resource_apply_base(grad, var)

    def _resource_apply_sparse(self, grad, var, indices, **kwargs):
        return self._resource_apply_base(grad, var, indices)

    def get_config(self):
        conf1 = super(AdamW, self).get_config()
        conf1.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'layer_norm': self.lnorm,
            'step': self.step,
            'decaying_rate': self.drate,
            'epsilon': self.epsilon})
        return conf1


# class LRS(keras.optimizers.schedules.LearningRateSchedule):
#     """The learning rate schedule of Adam optimizer"""
#     def __init__(self, dim, warmup=4000):
#         super(LRS, self).__init__()
#         self.dim, self.warmup = tf.cast(dim, tf.float32), warmup
#
#     def __call__(self, step):
#         a1 = tf.math.rsqrt(step)
#         a2 = step*(self.warmup**-1.5)
#         return tf.math.rsqrt(self.dim)*tf.math.minimum(a1, a2)


"""Model tools"""


class Tokenizer:
    """The text tokenizer"""
    def __init__(self):
        self.vocab = {}
        self.character = [
            [0x4E00, 0x9FFF], [0x3400, 0x4DBF], [0x20000, 0x2A6DF], [0x2A700, 0x2B73F], [0x2B740, 0x2B81F],
            [0x2B820, 0x2CEAF], [0xF900, 0xFAFF], [0x2F800, 0x2FA1F]]

    def loading(self, path, encoding='utf-8'):
        with open(path, encoding=encoding) as f1:
            for i1, j1 in enumerate(f1):
                self.vocab[j1.strip()] = i1

    def separating(self, text):
        text1, char1 = [], False

        for c1 in text:
            for i1 in self.character:
                if i1[0] <= ord(c1) <= i1[1]:
                    char1 = True
                    break

            text1, char1 = text1+[' ', c1, ' '] if char1 else text1+[c1], False

        return ''.join(text1).strip().split()

    def encoding(self, a, b=None, maxlen=128):
        a1 = ['[CLS]']+self.separating(a)+['[SEP]']
        b1 = self.separating(b)+['[SEP]'] if b is not None else []
        segm1 = ([0]*len(a1)+[1]*len(b1))[:maxlen]
        a1 = (a1+b1)[:maxlen]
        segm1 = segm1+[segm1[-1]]*(maxlen-len(segm1))
        mask1 = [0]*len(a1)+[1]*(maxlen-len(a1))
        a1 = a1+['[PAD]']*(maxlen-len(a1))
        return [self.vocab.get(i1, self.vocab['[UNK]']) for i1 in a1], segm1, mask1


"""BERT tests"""


if __name__ == '__main__':
    # toke_1 = Tokenizer()
    # toke_1.loading('./models/bert_base_ch/vocab.txt')
    # bert_1 = BERT('./models/bert_base_ch/bert_config.json', False)
    # bert_1.loading('./models/bert_base_ch/bert_model.ckpt')
    toke_1 = Tokenizer()
    toke_1.loading('./models/albert_base_ch/vocab_chinese.txt')
    bert_1 = BERT('./models/albert_base_ch/albert_config.json', True)
    bert_1.loading('./models/albert_base_ch/model.ckpt-best')
    text_1, segm_1, mask_1 = toke_1.encoding('我是头猪。', None, 128)
    sequ_1 = bert_1.propagating(tf.constant([text_1]), tf.constant([segm_1]), tf.constant([mask_1]))[:, :7, :]
