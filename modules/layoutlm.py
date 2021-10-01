# -*- coding: utf-8 -*-

"""
Implementation: Model of LayoutLMv2 and LayoutXLM (not completed)
References: https://arxiv.org/abs/2012.14740, https://huggingface.co/transformers/model_doc/layoutlmv2.html

"""

import re
import torch
import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras as keras


def w_initializing(param=0.02):
    return keras.initializers.TruncatedNormal(stddev=param)


def gelu_activating(x):
    return 0.5*x*(1.0+tf.math.erf(x/tf.math.sqrt(2.0)))


class Bottleneck(keras.layers.Layer):
    def __init__(self, bn, c, stride, group=8, ds=False, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.pad1 = keras.layers.ZeroPadding2D(1)
        self.conv1 = keras.layers.Conv2D(c, 1, 1, use_bias=False, name=bn+'conv1')
        self.conv2 = keras.layers.Conv2D(c, 3, stride, 'VALID', groups=group, use_bias=False, name=bn+'conv2')
        self.conv3 = keras.layers.Conv2D(c, 1, 1, use_bias=False, name=bn+'conv3')
        self.conv4 = keras.layers.Conv2D(c, 1, stride, use_bias=False, name=bn+'shortcut') if ds else None
        self.bn1 = keras.layers.BatchNormalization(3, 0.1, 1e-5, name=bn+'conv1.norm')
        self.bn2 = keras.layers.BatchNormalization(3, 0.1, 1e-5, name=bn+'conv2.norm')
        self.bn3 = keras.layers.BatchNormalization(3, 0.1, 1e-5, name=bn+'conv3.norm')
        self.bn4 = keras.layers.BatchNormalization(3, 0.1, 1e-5, name=bn+'shortcut.norm') if ds else None

    def propagating(self, x, training):
        x1 = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x1 = tf.nn.relu(self.bn2(self.conv2(self.pad1(x1)), training=training))
        x1 = self.bn3(self.conv3(x1), training=training)
        x2 = self.bn4(self.conv4(x), training=training) if self.conv4 is not None else x
        return tf.nn.relu(x1+x2)


class BlockFPN(keras.layers.Layer):
    def __init__(self, bn, ln, c=256, **kwargs):
        super(BlockFPN, self).__init__(**kwargs)
        self.pad1 = keras.layers.ZeroPadding2D(1)
        self.conv1 = keras.layers.Conv2D(c, 1, name=bn+'fpn_lateral{}'.format(ln))
        self.conv2 = keras.layers.Conv2D(c, 3, name=bn+'fpn_output{}'.format(ln))

    def propagating(self, x, up=None):
        x1 = self.conv1(x) if up is None else self.conv1(x)+tf.image.resize(up, tf.shape(x)[1:3], 'nearest')
        return x1, self.conv2(self.pad1(x1))


class ResNet(keras.layers.Layer):
    def __init__(self, bn, small=False, group=32, size=7, dim=256, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.mean, self.std = tf.constant([103.530, 116.280, 123.675]), tf.constant([57.375, 57.120, 58.395])
        self.pad1 = keras.layers.ZeroPadding2D(3)
        self.pad2 = keras.layers.ZeroPadding2D(1)
        self.conv = keras.layers.Conv2D(64, 7, 2, 'VALID', use_bias=False, name=bn+'bottom_up.stem.conv1')
        self.bn = keras.layers.BatchNormalization(3, 0.1, 1e-5, name=bn+'bottom_up.stem.conv1.norm')
        self.pool1 = keras.layers.MaxPool2D(3, 2, 'VALID')
        self.pool2 = tfa.layers.AdaptiveAveragePooling2D((size, size))
        self.fpn = [BlockFPN(bn, i1, dim) for i1 in [5, 4, 3, 2]]
        self.size, self.dim, bn = size, dim, bn+'bottom_up.res{}.{}.'
        self.block = [
            [Bottleneck(bn.format(2, i1), 256, 1, group, not i1) for i1 in range(3)],
            [Bottleneck(bn.format(3, i1), 512, 1 if i1 else 2, group, not i1) for i1 in range(4)],
            [Bottleneck(bn.format(4, i1), 1024, 1 if i1 else 2, group, not i1) for i1 in range(6 if small else 23)],
            [Bottleneck(bn.format(5, i1), 2048, 1 if i1 else 2, group, not i1) for i1 in range(3)]]

    def mapping(self, batch):
        x1 = (keras.backend.arange(0, 1000 * (self.size+1), 1000)//self.size)[tf.newaxis, :]
        p1, p2 = tf.repeat(x1[:, :-1], self.size, 0), tf.repeat(x1[:, 1:], self.size, 0)
        b1 = tf.stack([p1, tf.transpose(p1, [1, 0]), p2, tf.transpose(p2, [1, 0])], -1)
        return tf.repeat(tf.reshape(b1, [1, -1, 4]), batch, 0)

    def propagating(self, x, training=False):
        x1, b1 = self.bn(self.conv(self.pad1((x-self.mean)/self.std)), training=training), tf.shape(x)[0]
        x1, h1, r1, out1 = self.pool1(self.pad2(tf.nn.relu(x1))), None, [], [None]*len(self.block)

        for i1 in range(len(self.block)):
            for j1 in range(len(self.block[i1])):
                x1 = self.block[i1][j1].propagating(x1, training)
                r1 = (r1+[x1]) if j1 == len(self.block[i1])-1 else r1

        for i1 in range(len(self.fpn)):
            h1, out1[-i1-1] = self.fpn[i1].propagating(r1[-i1-1], h1)

        return out1, tf.reshape(self.pool2(out1[0]), [-1, self.size*self.size, self.dim]), self.mapping(b1)


class Attention(keras.layers.Layer):
    def __init__(self, bname, head, size, attdrop=0., drop=0., eps=1e-6, ninf=-1e4, fast=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.head, self.size, self.dim, self.ninf, self.fast = head, size, head*size, ninf, fast
        self.dense = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+'output.dense')
        self.norm = keras.layers.LayerNormalization(-1, eps, name=bname+'output.LayerNorm')
        self.attdrop = keras.layers.Dropout(attdrop)
        self.drop = keras.layers.Dropout(drop)

        if fast:
            self.wq = keras.layers.Dense(self.dim*3, None, False, w_initializing(), name=bname+'self.qkv_linear')
            self.qb = self.add_weight(bname+'self.q_bias', (1, 1, self.dim), None, w_initializing())
            self.vb = self.add_weight(bname+'self.v_bias', (1, 1, self.dim), None, w_initializing())
        else:
            self.wq = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+'self.query')
            self.wk = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+'self.key')
            self.wv = keras.layers.Dense(self.dim, None, True, w_initializing(), name=bname+'self.value')

    def transposing(self, x):
        return tf.transpose(tf.reshape(x, [-1, tf.shape(x)[1], self.head, self.size]), [0, 2, 1, 3])

    def masking(self, mask):
        return tf.cast(mask[:, tf.newaxis, tf.newaxis, :], tf.float32)*self.ninf

    def calculating(self, q, k, v, mask, rel, sp, training):
        a1 = tf.matmul(self.transposing(q), self.transposing(k), transpose_b=True)
        a1 = a1/tf.math.sqrt(tf.cast(self.size, tf.float32))
        a1 = a1+(0. if rel is None else rel)+(0. if sp is None else sp)
        a1 = tf.nn.softmax(a1+self.masking(mask) if mask is not None else a1, axis=-1)
        return tf.matmul(self.attdrop(a1, training=training), self.transposing(v)), a1

    def propagating(self, x, mask=None, rel=None, sp=None, training=False):
        q1, k1, v1 = (tf.split(self.wq(x), 3, 2)) if self.fast else (self.wq(x), self.wk(x), self.wv(x))
        q1, v1 = q1+self.qb if self.fast else q1, v1+self.vb if self.fast else v1
        x1, a1 = self.calculating(q1, k1, v1, mask, rel, sp, training)
        x1 = tf.reshape(tf.transpose(x1, [0, 2, 1, 3]), [-1, tf.shape(x1)[2], self.dim])
        return self.norm(x+self.drop(self.dense(x1), training=training)), a1


class TransEncoder(keras.layers.Layer):
    def __init__(self, bname, head, size, dff, attdrop=0., drop=0., eps=1e-6, fast=False, act=None, **kwargs):
        super(TransEncoder, self).__init__(**kwargs)
        self.att = Attention(bname+'attention.', head, size, attdrop, drop, eps, fast=fast)
        self.norm = keras.layers.LayerNormalization(-1, eps, name=bname+'output.LayerNorm')
        self.dense1 = keras.layers.Dense(dff, act, True, w_initializing(), name=bname+'intermediate.dense')
        self.dense2 = keras.layers.Dense(int(head*size), None, True, w_initializing(), name=bname+'output.dense')
        self.drop = keras.layers.Dropout(drop)

    def propagating(self, x, mask=None, rel=None, sp=None, training=False):
        x1, a1 = self.att.propagating(x, mask, rel, sp, training)
        return self.norm(x1+self.drop(self.dense2(self.dense1(x1)), training=training))


class Embedding(keras.layers.Layer):
    def __init__(self, bn, maxlen, maxpos, vocab, seg, dim=768, coord=128, drop=.1, eps=1e-6, visual=False, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.ve = self.add_weight(bn+'visual_segment_embedding', [dim], None, w_initializing()) if visual else None
        self.norm1, bn = keras.layers.LayerNormalization(-1, 1e-6, name=bn+'visual_LayerNorm'), bn+'embeddings.'
        self.norm2 = keras.layers.LayerNormalization(-1, eps, name=bn+'LayerNorm')
        self.emb1 = keras.layers.Embedding(maxpos, coord, w_initializing(), name=bn+'x_position_embeddings')
        self.emb2 = keras.layers.Embedding(maxpos, coord, w_initializing(), name=bn+'y_position_embeddings')
        self.emb3 = keras.layers.Embedding(maxpos, (dim-4*coord)//2, w_initializing(), name=bn+'h_position_embeddings')
        self.emb4 = keras.layers.Embedding(maxpos, (dim-4*coord)//2, w_initializing(), name=bn+'w_position_embeddings')
        self.emb5 = self.add_weight(bn+'position_embeddings.weight', (maxlen, dim), None, w_initializing())
        self.emb6 = keras.layers.Embedding(vocab, dim, w_initializing(), name=bn+'word_embeddings')
        self.emb7 = keras.layers.Embedding(seg, dim, w_initializing(), name=bn+'token_type_embeddings')
        self.drop1 = keras.layers.Dropout(drop)
        self.drop2 = keras.layers.Dropout(drop)

    def calculating(self, box):
        p1, p2 = self.emb1(box[:, :, 0]), self.emb1(box[:, :, 2])
        p3, p4 = self.emb2(box[:, :, 1]), self.emb2(box[:, :, 3])
        p5, p6 = self.emb3(box[:, :, 3]-box[:, :, 1]), self.emb4(box[:, :, 2]-box[:, :, 0])
        return tf.concat([p1, p3, p2, p4, p5, p6], -1)

    def propagating(self, image, imagebox, text, seg, textbox, training=False):
        v1 = image+tf.slice(self.emb5, [0, 0], [tf.shape(image)[1], -1])+self.calculating(imagebox)
        t1 = self.emb6(text)+tf.slice(self.emb5, [0, 0], [tf.shape(text)[1], -1])+self.calculating(textbox)
        v1 = self.drop1(self.norm1(v1 if self.ve is None else v1+self.ve), training=training)
        t1 = self.drop2(self.norm2(t1+self.emb7(seg)), training=training)
        return tf.concat([t1, v1], 1)


class LayoutLM(keras.layers.Layer):
    def __init__(self, model, config=None, **kwargs):
        super(LayoutLM, self).__init__(**kwargs)
        self.param, n1 = self.checking(model) if config is None else config, 'layoutlmv2.'
        self.vproj = keras.layers.Dense(self.param['hidden_size'], name='layoutlmv2.visual_proj')
        self.b1 = keras.layers.Dense(self.param['num_attention_heads'], None, False, name=n1+'encoder.rel_pos_bias')
        self.b2 = keras.layers.Dense(self.param['num_attention_heads'], None, False, name=n1+'encoder.rel_pos_x_bias')
        self.b3 = keras.layers.Dense(self.param['num_attention_heads'], None, False, name=n1+'encoder.rel_pos_y_bias')
        self.pool = keras.layers.Dense(self.param['hidden_size'], None, True, name=n1+'pooler.dense')
        self.resnet = ResNet(
            bn=n1+'visual.backbone.',
            size=self.param['image_feature_pool_shape'][0],
            dim=self.param['image_feature_pool_shape'][2])
        self.emb = Embedding(
            n1,
            self.param['max_position_embeddings'],
            self.param['max_2d_position_embeddings'],
            self.param['vocab_size'],
            self.param['type_vocab_size'],
            self.param['hidden_size'],
            self.param['coordinate_size'],
            self.param['hidden_dropout_prob'],
            self.param['layer_norm_eps'],
            self.param['has_visual_segment_embedding'])
        self.encoder = [TransEncoder(
            n1+'encoder.layer.{}.'.format(i1),
            self.param['num_attention_heads'],
            self.param['hidden_size']//self.param['num_attention_heads'],
            self.param['intermediate_size'],
            self.param['attention_probs_dropout_prob'],
            self.param['hidden_dropout_prob'],
            self.param['layer_norm_eps'],
            self.param['fast_qkv'],
            self.param['hidden_act']) for i1 in range(self.param['num_hidden_layers'])]
        self.rpl = {'/': '.', 'kernel': 'weight', 'gamma': 'weight', 'beta': 'bias', 'mov': 'runn', 'variance': (
            'var'), 's/embeddings': 's.weight'}

    @staticmethod
    def checking(config):
        return {
            'layoutlmv2-base': {
                'has_visual_segment_embedding': False,
                'has_relative_attention_bias': True,
                'has_spatial_attention_bias': True,
                'fast_qkv': True,
                'hidden_size': 768,
                'num_attention_heads': 12,
                'num_hidden_layers': 12,
                'max_rel_2d_pos': 256,
                'max_rel_pos': 128,
                'rel_2d_pos_bins': 64,
                'rel_pos_bins': 32,
                'intermediate_size': 3072,
                'layer_norm_eps': 1e-12,
                'attention_probs_dropout_prob': 0.1,
                'hidden_dropout_prob': 0.1,
                'hidden_act': 'gelu',
                'coordinate_size': 128,
                'max_2d_position_embeddings': 1024,
                'max_position_embeddings': 512,
                'image_feature_pool_shape': [7, 7, 256],
                'type_vocab_size': 2,
                'vocab_size': 30522},
            'layoutlmv2-large': {
                'has_visual_segment_embedding': False,
                'has_relative_attention_bias': True,
                'has_spatial_attention_bias': True,
                'fast_qkv': False,
                'hidden_size': 1024,
                'num_attention_heads': 16,
                'num_hidden_layers': 24,
                'max_rel_2d_pos': 256,
                'max_rel_pos': 128,
                'rel_2d_pos_bins': 64,
                'rel_pos_bins': 32,
                'intermediate_size': 4096,
                'layer_norm_eps': 1e-12,
                'attention_probs_dropout_prob': 0.1,
                'hidden_dropout_prob': 0.1,
                'hidden_act': 'gelu',
                'coordinate_size': 171,
                'max_2d_position_embeddings': 1024,
                'max_position_embeddings': 512,
                'image_feature_pool_shape': [7, 7, 256],
                'type_vocab_size': 2,
                'vocab_size': 30522},
            'layoutxlm-base': {
                'has_visual_segment_embedding': True,
                'has_relative_attention_bias': False,
                'has_spatial_attention_bias': False,
                'fast_qkv': False,
                'hidden_size': 768,
                'num_attention_heads': 12,
                'num_hidden_layers': 12,
                'max_rel_2d_pos': 256,
                'max_rel_pos': 128,
                'rel_2d_pos_bins': 64,
                'rel_pos_bins': 32,
                'intermediate_size': 3072,
                'layer_norm_eps': 1e-12,
                'attention_probs_dropout_prob': 0.1,
                'hidden_dropout_prob': 0.1,
                'hidden_act': 'gelu',
                'coordinate_size': 128,
                'max_2d_position_embeddings': 1024,
                'max_position_embeddings': 514,
                'image_feature_pool_shape': [7, 7, 256],
                'type_vocab_size': 1,
                'vocab_size': 250002}}[config] if type(config) is str else config

    def loading(self, pth):
        _ = self.propagating(tf.ones((2, 224, 224, 3)), tf.zeros((2, 2), tf.int32), tf.zeros((2, 2, 4), tf.int32))
        r1, l1 = re.compile('|'.join(map(re.escape, self.rpl))), torch.load(pth)
        n1, t1 = [r1.sub((lambda x1: self.rpl[x1.group(0)]), i1.name[:-2]) for i1 in self.weights], []

        for i1 in n1:
            if len(l1[i1].shape) == 4:
                t1.append(tf.transpose(l1[i1].numpy(), [2, 3, 1, 0]))
            elif 'weight' in i1 and 'embed' not in i1 and len(l1[i1].shape) == 2:
                t1.append(tf.transpose(l1[i1].numpy(), [1, 0]))
            else:
                t1.append(l1[i1].numpy())

        keras.backend.batch_set_value(zip(self.weights, t1))

    def mapping(self, pos, sp=True):
        p1, b1 = tf.expand_dims(pos, -2)-tf.expand_dims(pos, -1), self.param['rel'+('_2d' if sp else '')+'_pos_bins']
        r1, buck1 = tf.cast(tf.math.less(-p1, 0), tf.int32)*(b1//2), b1//2
        n1, exac1, m1 = tf.math.abs(p1), buck1//2, self.param['max_rel'+('_2d' if sp else '')+'_pos']
        larg1 = tf.cast(tf.math.log(tf.cast(n1, tf.float32)/exac1)/tf.math.log(m1/exac1)*(buck1-exac1), tf.int32)
        return tf.one_hot(r1+tf.where(tf.less(n1, exac1), n1, tf.minimum(exac1+larg1, buck1-1)), b1)

    def calculating(self, pos, box):
        t1, t2 = self.param['has_relative_attention_bias'], self.param['has_spatial_attention_bias']
        r1 = tf.transpose(self.b1(self.mapping(pos, False)), [0, 3, 1, 2]) if t1 else None
        x1 = tf.transpose(self.b2(self.mapping(box[:, :, 0], True)), [0, 3, 1, 2]) if t2 else None
        y1 = tf.transpose(self.b3(self.mapping(box[:, :, 3], True)), [0, 3, 1, 2]) if t2 else None
        return r1, x1+y1 if t2 else None

    def propagating(self, x, text, box, seg=None, mask=None, training=False):
        s1, m1 = tf.zeros_like(text) if seg is None else seg, tf.zeros_like(text) if mask is None else mask
        list1, stat1, bbox1 = self.resnet.propagating(x, training=training)
        p1 = tf.repeat(keras.backend.arange(0, tf.shape(text)[1])[tf.newaxis, :], tf.shape(text)[0], 0)
        p2 = tf.repeat(keras.backend.arange(0, self.resnet.size**2)[tf.newaxis, :], tf.shape(text)[0], 0)
        x1 = self.emb.propagating(self.vproj(stat1), bbox1, text, s1, box, training)
        r1, r2 = self.calculating(tf.concat([p1, p2], 1), tf.concat([box, bbox1], 1))

        for i1 in range(len(self.encoder)):
            x1 = self.encoder[i1].propagating(x1, mask, r1, r2, training=training)

        return stat1, x1, tf.nn.tanh(self.pool(x1[:, 0, :]))
