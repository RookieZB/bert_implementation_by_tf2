# -*- coding: utf-8 -*-

"""
Title: My Implementation of Models by TensorFlow
Author: Chris Chen
Date: 2020/02/02
Update: 2020/07/31

"""

import os
import warnings
import unicodedata
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.python as ops

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


class InstanceNormalization(keras.layers.Layer):
    """Instance normalization layer"""
    def __init__(self, eps=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon, self.scale, self.offset = eps, None, None

    def build(self, size):
        self.scale = self.add_weight('scale', size[-1:], None, tf.random_normal_initializer(1., 0.02), trainable=True)
        self.offset = self.add_weight('offset', size[-1:], None, 'zeros', trainable=True)

    def propagating(self, x):
        m1, v1 = tf.nn.moments(x, [1, 2], None, True)
        return self.scale*(x-m1)*tf.math.rsqrt(v1+self.epsilon)+self.offset


class Attention(keras.layers.Layer):
    """Attention layer"""
    def __init__(self, bname, lname, head, size, attdrop, drop, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.head, self.size, self.ninf = head, size, -1e4
        self.wq = keras.layers.Dense(self.head*self.size, None, True, w_initializing(), name=bname+lname[0])
        self.wk = keras.layers.Dense(self.head*self.size, None, True, w_initializing(), name=bname+lname[1])
        self.wv = keras.layers.Dense(self.head*self.size, None, True, w_initializing(), name=bname+lname[2])
        self.dense = keras.layers.Dense(self.head*self.size, None, True, w_initializing(), name=bname+lname[3])
        self.norm = keras.layers.LayerNormalization(-1, 1e-6, name=bname+lname[4])
        self.attdrop = keras.layers.Dropout(attdrop)
        self.drop = keras.layers.Dropout(drop)

    def transposing(self, x):
        return tf.transpose(tf.reshape(x, [-1, x.shape[1], self.head, self.size]), [0, 2, 1, 3])

    def masking(self, mask):
        return tf.cast(mask[:, tf.newaxis, tf.newaxis, :], tf.float32)*self.ninf

    def calculating(self, q, k, v, training, mask):
        atte1 = tf.matmul(q, k, transpose_b=True)/tf.math.sqrt(tf.cast(self.size, tf.float32))
        atte1 = tf.nn.softmax(atte1+self.masking(mask) if mask is not None else atte1, axis=-1)
        return tf.matmul(self.attdrop(atte1, training=training), v), atte1

    def propagating(self, x, training, mask):
        q1 = self.transposing(self.wq(x))
        k1 = self.transposing(self.wk(x))
        v1 = self.transposing(self.wv(x))
        x1, a1 = self.calculating(q1, k1, v1, training, mask)
        x1 = tf.reshape(tf.transpose(x1, [0, 2, 1, 3]), [-1, x1.shape[2], self.head*self.size])
        return self.norm(x+self.drop(self.dense(x1), training=training)), a1


class TransEncoder(keras.layers.Layer):
    """Transformer encoder layer"""
    def __init__(self, bname, lname, head, size, dff, attdrop=0.1, drop=0.1, act='relu', **kwargs):
        super(TransEncoder, self).__init__(**kwargs)
        self.att = Attention(bname, lname, head, size, attdrop, drop)
        self.norm = keras.layers.LayerNormalization(-1, 1e-6, name=bname+lname[7])
        self.drop = keras.layers.Dropout(drop)
        self.dense1 = keras.layers.Dense(int(head*size), None, True, w_initializing(), name=bname+lname[6])
        self.dense2 = keras.layers.Dense(dff, act, True, w_initializing(), name=bname+lname[5])

    def propagating(self, x, training, mask):
        x1, _ = self.att.propagating(x, training, mask)
        x2 = self.drop(self.dense1(self.dense2(x1)), training=training)
        return self.norm(x1+x2)


class Embedding(keras.layers.Layer):
    """Embedding layer"""
    def __init__(self, bname, lname, voc, seg, dim, maxlen, drop=0.1, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.embedding = keras.layers.Embedding(voc, dim, w_initializing(), name=bname+lname[0])
        self.segemb = keras.layers.Embedding(seg, dim, w_initializing(), name=bname+lname[1])
        self.posemb = self.add_weight(bname+lname[2], (maxlen, dim), None, w_initializing())
        self.norm = keras.layers.LayerNormalization(-1, 1e-6, name=bname+lname[3])
        self.drop = keras.layers.Dropout(drop)

    def propagating(self, x, seg, training):
        e1 = self.embedding(x)+self.segemb(seg)+tf.slice(self.posemb, [0, 0], [x.shape[1], -1])
        return self.drop(self.norm(e1), training=training)


class Bottleneck(keras.layers.Layer):
    """Bottleneck layer"""
    def __init__(self, lname, size, stride, cin, cout, expansion, caxis, act, squeeze=True, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.div, self.cin, self.filter, self.stride, self.squeeze = 8, cin, cout, stride, squeeze
        self.exp = int(cin*expansion)
        self.factor = self.dividing(self.exp/4)
        self.act = {'hswish': hswish_activating, 'relu': tf.nn.relu}[act]
        self.conv1 = keras.layers.Conv2D(self.exp, 1, 1, 'same', use_bias=False, name=lname+'expand/')
        self.conv2 = keras.layers.DepthwiseConv2D(size, stride, 'same', use_bias=False, name=lname+'depthwise/')
        self.conv3 = keras.layers.Conv2D(cout, 1, 1, 'same', use_bias=False, name=lname+'project/')
        self.norm1 = keras.layers.BatchNormalization(caxis, 0.999, name=lname+'expand/BatchNorm/')
        self.norm2 = keras.layers.BatchNormalization(caxis, 0.999, name=lname+'depthwise/BatchNorm/')
        self.norm3 = keras.layers.BatchNormalization(caxis, 0.999, name=lname+'project/BatchNorm/')
        self.pool = keras.layers.GlobalAveragePooling2D()
        self.s1 = keras.layers.Conv2D(self.factor, 1, activation='relu', name=lname+'squeeze_excite/Conv/')
        self.s2 = keras.layers.Conv2D(self.exp, 1, activation=hsigmoid_activating, name=lname+'squeeze_excite/Conv_1/')

    def dividing(self, v, minv=None):
        valu1 = max(self.div if minv is None else minv, int(v+self.div/2)//self.div*self.div)
        return valu1+self.div if valu1 < 0.9*v else valu1

    def propagating(self, x, training):
        x1 = self.act(self.norm1(self.conv1(x), training=training)) if self.cin != self.exp else x
        x1 = self.act(self.norm2(self.conv2(x1), training=training))

        if self.squeeze:
            h1 = tf.reshape(self.pool(x1), [-1, 1, 1, self.exp])
            x1 = x1*self.s2(self.s1(h1))

        x1 = self.norm3(self.conv3(x1), training=training)
        return x+x1 if self.cin == self.filter and self.stride == 1 else x1


class CRF(keras.layers.Layer):
    """Conditional random field"""
    def __init__(self, dim, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.transition = self.add_weight('crf/transition', (dim, dim), None, 'glorot_uniform')

    def scanning(self, _state, _inputs):
        _state = tf.expand_dims(_state, 2)
        tran1 = tf.expand_dims(self.transition, 0)
        return _inputs+tf.reduce_logsumexp(_state+tran1, 1)

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
        path1 = np.zeros_like(batch)
        stat1 = np.zeros_like(batch)
        stat1[:, 0] = batch[:, 0]

        for i1 in range(1, np.max(mask1)):
            s1 = np.expand_dims(stat1[:, i1-1], 2)+tran1
            stat1[:, i1] = np.max(s1, 1)+batch[:, i1]
            path1[:, i1] = np.argmax(s1, 1)

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
        self.replacement = {
            'bert/embeddings/word_embeddings/embeddings': 'bert/embeddings/word_embeddings',
            'bert/embeddings/token_type_embeddings/embeddings': 'bert/embeddings/token_type_embeddings',
            'electra/embeddings/word_embeddings/embeddings': 'electra/embeddings/word_embeddings',
            'electra/embeddings/token_type_embeddings/embeddings': 'electra/embeddings/token_type_embeddings'}

        if self.mode not in ['cls', 'seq', 'mlm']:
            raise Exception('Unrecognized mode "{}".'.format(self.mode))

        if self.model in ['bert', 'albert', 'electra', 'roberta'] or self.model.split('*')[0] in ['bert', 'albert']:
            self.cat = 1 if self.model.split('*')[0] == 'albert' else 0
            self.hd = 'bert' if self.model in ['albert', 'roberta'] else self.model.split('*')[-1]
            self.namep = self.hd+('/encoder/embedding_hidden_mapping_in' if self.cat else '/embeddings_project')
            self.namee = ['/word_embeddings', '/token_type_embeddings', '/position_embeddings', '/LayerNorm']
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
            self.namel = [
                'cls/predictions/transform/dense',
                'cls/predictions/transform/LayerNorm',
                'cls/predictions/output_bias']
            self.embedding = Embedding(
                self.hd+'/embeddings',
                self.namee,
                self.param['vocab_size'],
                self.param['type_vocab_size'],
                self.embsize,
                self.param['max_position_embeddings'],
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
                self.outbias = self.add_weight(self.namel[2], self.param['vocab_size'], initializer='zeros')

        else:
            raise Exception('Unrecognized model "{}".'.format(self.model))

    def loading(self, ckpt):
        _ = self.propagating(tf.ones((2, 2)), tf.zeros((2, 2)), tf.zeros((2, 2)))
        name1 = [self.replacement.get(i1.name[:-2], i1.name[:-2]) for i1 in self.weights]
        keras.backend.batch_set_value(zip(self.weights, [tf.train.load_variable(ckpt, i1) for i1 in name1]))

    def propagating(self, x, seg, mask, training=False):
        x1 = self.embedding.propagating(x, seg, training)
        x1 = self.projection(x1) if self.projection is not None else x1

        for i1 in range(self.param['num_hidden_layers']):
            x1 = self.encoder[0 if self.cat == 1 else i1].propagating(x1, training, mask)

        if self.mode == 'mlm':
            x2 = self.norm(self.dense(x1))
            x2 = tf.nn.bias_add(tf.matmul(x2, self.embedding.embedding.embeddings, transpose_b=True), self.outbias)
            return tf.nn.softmax(x2), x1
        else:
            return x1[:, 0, :] if self.mode == 'cls' else x1


class MobileNet(keras.layers.Layer):
    """MobileNet V3 model"""
    def __init__(self, alpha, caxis, cate, act='hswish', param=None, **kwargs):
        super(MobileNet, self).__init__(**kwargs)
        self.act = hswish_activating if act == 'hswish' else act
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
            (96, 5, 1, 6, 'hswish', True)] if param is None else param
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
        name1 = [i1.name[:-2].replace('kernel', 'weights').replace('bias', 'biases') for i1 in self.weights]
        keras.backend.batch_set_value(zip(self.weights, [tf.train.load_variable(ckpt, i1) for i1 in name1]))

    def propagating(self, x, training=False):
        x1 = self.act(self.norm1(self.conv1(x), training=training))

        for i1 in self.encoder:
            x1 = i1.propagating(x1, training)

        x1 = self.act(self.norm2(self.conv2(x1), training=training))
        x1 = tf.reshape(self.pool(x1), (-1, 1, 1, x1.shape[-1]))
        return tf.nn.softmax(self.conv4(self.act(self.conv3(x1))))


"""Optimizers"""


class AdamW(keras.optimizers.Optimizer):
    """AdamW optimizer"""
    def __init__(self, step, lrate=1e-3, b1=0.9, b2=0.999, drate=1e-2, lmode=0, ldecay=None, name='AdamW', **kwargs):
        super(AdamW, self).__init__(name, **kwargs)
        self.step, self.drate, self.lmode, self.ldecay, self.epsilon = step, drate, lmode, ldecay, 1e-6
        self.spec = ['LayerNorm', 'bias']
        self._set_hyper('learning_rate', lrate)
        self._set_hyper('beta_1', b1)
        self._set_hyper('beta_2', b2)

        if lmode not in [0, 1, 2]:
            raise Exception('Unrecognized layer mode "{}".'.format(self.lmode))

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
        step1 = tf.cast(self.iterations+1, var_dtype)
        beta1 = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta2 = tf.identity(self._get_hyper('beta_2', var_dtype))
        rate1 = self._rate_sch(apply_state[(var_device, var_dtype)]['lr_t'], step1, self.step+1)
        apply_state[(var_device, var_dtype)].update(dict(
            lr=rate1,
            epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
            beta_1=beta1,
            beta_1_minus=1-beta1,
            beta_2=beta2,
            beta_2_minus=1-beta2))

    def _resource_apply_base(self, grad, var, indices=None, apply_state=None):
        m1 = self.get_slot(var, 'm')
        v1 = self.get_slot(var, 'v')
        spec1 = False
        coef1 = ((apply_state or {}).get((var.device, var.dtype.base_dtype)) or self._fallback_apply_state(
            var.device, var.dtype.base_dtype))

        if indices is None:
            m2 = m1.assign(coef1['beta_1']*m1+coef1['beta_1_minus']*grad, self._use_locking)
            v2 = v1.assign(coef1['beta_2']*v1+coef1['beta_2_minus']*grad*grad, self._use_locking)
        else:
            m2 = m1.assign(coef1['beta_1']*m1, self._use_locking)
            v2 = v1.assign(coef1['beta_2']*v1, self._use_locking)

            with tf.control_dependencies([m2, v2]):
                m2 = self._resource_scatter_add(m1, indices, coef1['beta_1_minus']*grad)
                v2 = self._resource_scatter_add(v1, indices, coef1['beta_2_minus']*grad*grad)

        for item1 in self.spec:
            if item1 in var.name:
                spec1 = True
                break

        u1 = m2/(tf.sqrt(v2)+coef1['epsilon']) if spec1 else m2/(tf.sqrt(v2)+coef1['epsilon'])+self.drate*var
        rati1 = 1.0

        if self.lmode == 1 and not spec1:
            n1 = tf.norm(var, 2)
            n2 = tf.norm(u1, 2)
            rati1 = tf.where(tf.greater(n1, 0.), tf.where(tf.greater(n2, 0.), n1/n2, 1.), 1.)

        if self.lmode == 2 and not spec1:
            for item2 in self.ldecay.keys():
                if item2 in var.name:
                    rati1 = self.ldecay[item2]
                    break

        return tf.group(*[var.assign_sub(rati1*coef1['lr']*u1, self._use_locking), m2, v2])

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
    def __init__(self, step, lrate=1e-3, b1=0.9, b2=0.999, drate=1e-2, name='AdamW', **kwargs):
        super(AdamWV2, self).__init__(lrate, b1, b2, name=name, **kwargs)
        self.step, self.drate = step, drate
        self.spec = ['LayerNorm', 'bias']

    @staticmethod
    def _rate_sch(rate, step, total):
        warm1 = total*0.1
        return tf.where(step < warm1, rate*step/warm1, rate*(total-step)/(total-warm1))

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamWV2, self)._prepare_local(var_device, var_dtype, apply_state)
        step1 = tf.cast(self.iterations+1, var_dtype)
        rate1 = self._rate_sch(apply_state[(var_device, var_dtype)]['lr_t'], step1, self.step+1)
        apply_state[(var_device, var_dtype)].update(dict(lr=rate1))

    def _resource_apply_base(self, grad, var, indices=None, apply_state=None):
        m1 = self.get_slot(var, 'm')
        v1 = self.get_slot(var, 'v')
        spec1 = False
        coef1 = ((apply_state or {}).get((var.device, var.dtype.base_dtype)) or self._fallback_apply_state(
            var.device, var.dtype.base_dtype))

        for item1 in self.spec:
            if item1 in var.name:
                spec1 = True
                break

        d1 = coef1['lr']*var*self.drate
        deca1 = tf.no_op if spec1 else var.assign_sub(d1, use_locking=self._use_locking)

        if indices is None:
            with tf.control_dependencies([deca1]):
                return ops.training.training_ops.resource_apply_adam(
                    var.handle,
                    m1.handle,
                    v1.handle,
                    coef1['beta_1_power'],
                    coef1['beta_2_power'],
                    coef1['lr'],
                    coef1['beta_1_t'],
                    coef1['beta_2_t'],
                    coef1['epsilon'],
                    grad,
                    use_locking=self._use_locking)

        m2 = m1.assign(coef1['beta_1_t']*m1, self._use_locking)
        v2 = v1.assign(coef1['beta_2_t']*v1, self._use_locking)

        with tf.control_dependencies([m2, v2, deca1]):
            m2 = self._resource_scatter_add(m1, indices, coef1['one_minus_beta_1_t']*grad)
            v2 = self._resource_scatter_add(v1, indices, coef1['one_minus_beta_2_t']*grad*grad)
            u1 = coef1['lr']*m2/(tf.sqrt(v2)+coef1['epsilon'])
            return tf.group(*[var.assign_sub(u1, self._use_locking), m2, v2])

    def _resource_apply_dense(self, grad, var, apply_state=None, **kwargs):
        return self._resource_apply_base(grad, var, None, apply_state)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None, **kwargs):
        return self._resource_apply_base(grad, var, indices, apply_state)

    def get_config(self):
        conf1 = super(AdamWV2, self).get_config()
        conf1.update({'decaying_rate': self.drate, 'step': self.step})
        return conf1


class MovingAverage(object):
    """Exponential moving average strategy"""
    def __init__(self, var, decay=0.99):
        self.decay = decay
        self.var = [keras.backend.zeros(w.shape) for w in var]
        keras.backend.batch_set_value(zip(self.var, keras.backend.batch_get_value(var)))
        self.updating(var)

    def updating(self, var):
        for w1, w2 in zip(self.var, var):
            keras.backend.moving_average_update(w1, w2, self.decay)

    def setting(self, var):
        keras.backend.batch_set_value(zip(var, keras.backend.batch_get_value(self.var)))


"""Tools"""


class Tokenizer:
    """Simple BERT tokenizer"""
    def __init__(self):
        self.vocab, self.character = {}, [
            [0x4E00, 0x9FFF], [0x3400, 0x4DBF], [0x20000, 0x2A6DF], [0x2A700, 0x2B73F], [0x2B740, 0x2B81F],
            [0x2B820, 0x2CEAF], [0xF900, 0xFAFF], [0x2F800, 0x2FA1F], [33, 64], [91, 96], [123, 126]]

    def loading(self, path, encoding='utf-8'):
        for i1, j1 in enumerate(open(path, encoding=encoding)):
            self.vocab[j1.strip()] = i1

    def separating(self, text):
        text1 = []
        char1 = False

        for c1 in text.lower():
            for i1 in self.character:
                if i1[0] <= ord(c1) <= i1[1] or unicodedata.category(c1).startswith('P'):
                    char1 = True
                    break

            text1 = text1+[' ', c1, ' '] if char1 else text1+[c1]
            char1 = False

        text2 = ''.join(text1).replace('[ mask ]', '[MASK]').strip().split()
        return text2, len(text2)

    def truncating(self, a, b, maxlen):
        a1, l1 = self.separating(a)
        b1, l2 = self.separating(b) if b is not None else ([], 0)
        a2 = a1[:min(l1, int(np.ceil(maxlen/2)))] if l1 < l2 else a1[:max(maxlen-l2, int(np.ceil(maxlen/2)))]
        b2 = b1[:min(l2, int(np.floor(maxlen/2)))] if l2 < l1 else b1[:max(maxlen-l1, int(np.floor(maxlen/2)))]
        return a2, b2

    def encoding(self, a, b=None, maxlen=64, length=False):
        a1, b1 = self.truncating(a, b, maxlen-2 if b is None else maxlen-3)
        a1 = ['[CLS]']+a1+['[SEP]']
        b1 = b1+['[SEP]'] if b is not None else b1
        leng1 = len(a1)+len(b1)
        sent1 = [self.vocab.get(i1, self.vocab['[UNK]']) for i1 in (a1+b1+['[PAD]']*(maxlen-leng1))]
        segm1 = [0]*len(a1)+[1]*len(b1)+[0 if b is None else 1]*(maxlen-leng1)
        mask1 = [0]*leng1+[1]*(maxlen-leng1)
        return (sent1, segm1, mask1, leng1) if length else (sent1, segm1, mask1)


"""Tests"""


def bert_testing(sentence=None, mlm=False):
    """Test of BERT"""
    toke1 = Tokenizer()
    toke1.loading('./models/bert_base_ch/vocab.txt')
    voca1 = list(toke1.vocab.keys())
    bert1 = BERT('./models/bert_base_ch/bert_config.json', 'bert', 'mlm' if mlm else 'seq')
    bert1.loading('./models/bert_base_ch/bert_model.ckpt')
    text1, segm1, mask1 = toke1.encoding('感觉[MASK]就像一只猪。' if sentence is None else sentence)
    pred1 = bert1.propagating(np.array([text1]), np.array([segm1]), np.array([mask1]))
    pred1 = np.array(np.argmax(pred1[0], axis=-1)[0]) if mlm else pred1[:, :text1.index(0), :]
    print(''.join([voca1[pred1[i1]] for i1 in range(1, text1.index(0)-1)]) if mlm else pred1)


def mobile_testing(image=None):
    """Test of MobileNet"""
    imag1 = tf.image.decode_jpeg(tf.io.read_file('./models/v3_small_float/panda.jpg' if image is None else image), 3)
    imag1 = tf.expand_dims(tf.image.resize(tf.cast(imag1, tf.float32)/128.-1., [224, 224]), 0)
    mnet1 = MobileNet(1, -1, 1001)
    mnet1.loading('./models/v3_small_float/model-388500')
    inpu1 = keras.Input(shape=(224, 224, 3))
    mode1 = keras.Model(inputs=inpu1, outputs=mnet1.propagating(inpu1))
    print(np.flipud(np.argsort(mode1.predict(imag1)[0, 0, 0, :]))[:5])
