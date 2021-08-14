# -*- coding: utf-8 -*-

"""
DETR Implementation (not completed)
https://github.com/facebookresearch/detr

"""

import re
import torch
import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras as keras


class Bottleneck(keras.layers.Layer):
    def __init__(self, ln, c, stride, exp=4, ds=False, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.pad1 = keras.layers.ZeroPadding2D(1)
        self.conv1 = keras.layers.Conv2D(c, 1, 1, use_bias=False, name=ln+'conv1')
        self.conv2 = keras.layers.Conv2D(c, 3, stride, 'VALID', use_bias=False, name=ln+'conv2')
        self.conv3 = keras.layers.Conv2D(c*exp, 1, 1, use_bias=False, name=ln+'conv3')
        self.conv4 = keras.layers.Conv2D(c*exp, 1, stride, use_bias=False, name=ln+'downsample.0') if ds else None
        self.bn1 = keras.layers.BatchNormalization(3, 0.1, 1e-5, name=ln+'bn1')
        self.bn2 = keras.layers.BatchNormalization(3, 0.1, 1e-5, name=ln+'bn2')
        self.bn3 = keras.layers.BatchNormalization(3, 0.1, 1e-5, name=ln+'bn3')
        self.bn4 = keras.layers.BatchNormalization(3, 0.1, 1e-5, name=ln+'downsample.1') if ds else None

    def propagating(self, x, training):
        x1 = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x1 = tf.nn.relu(self.bn2(self.conv2(self.pad1(x1)), training=training))
        x1 = self.bn3(self.conv3(x1), training=training)
        x2 = self.bn4(self.conv4(x), training=training) if self.conv4 is not None else x
        return tf.nn.relu(x1+x2)


class ResNet(keras.layers.Layer):
    def __init__(self, ln, small=True, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.pad1 = keras.layers.ZeroPadding2D(3)
        self.pad2 = keras.layers.ZeroPadding2D(1)
        self.conv1 = keras.layers.Conv2D(64, 7, 2, 'VALID', use_bias=False, name=ln+'conv1')
        self.bn1 = keras.layers.BatchNormalization(3, 0.1, 1e-5, name=ln+'bn1')
        self.pool1, ln = keras.layers.MaxPool2D(3, 2, 'VALID'), ln+'layer{}.{}.'
        self.block1 = [
            [Bottleneck(ln.format(1, i1), 64, 1, ds=not i1) for i1 in range(3)],
            [Bottleneck(ln.format(2, i1), 128, 1 if i1 else 2, ds=not i1) for i1 in range(4)],
            [Bottleneck(ln.format(3, i1), 256, 1 if i1 else 2, ds=not i1) for i1 in range(6 if small else 23)],
            [Bottleneck(ln.format(4, i1), 512, 1 if i1 else 2, ds=not i1) for i1 in range(3)]]

    def propagating(self, x, training=False):
        x1 = self.bn1(self.conv1(self.pad1(x)), training=training)
        x1, r1 = self.pool1(self.pad2(tf.nn.relu(x1))), []

        for i1 in range(len(self.block1)):
            for j1 in range(len(self.block1[i1])):
                x1 = self.block1[i1][j1].propagating(x1, training)

            r1.append(x1)

        return r1


class PosEmbedding(keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super(PosEmbedding, self).__init__(**kwargs)
        self.dim, self.temp, self.scale = dim, 10000, np.pi*2

    def propagating(self, mask):
        d1 = self.temp**(2*(tf.cast(np.arange(self.dim), tf.float32)//2)/self.dim)
        e1, e2 = tf.math.cumsum(1.-mask, 1), tf.math.cumsum(1.-mask, 2)
        e1, e2 = e1/(e1[:, -1:, :]+1e-6)*self.scale, e2/(e2[:, :, -1:]+1e-6)*self.scale
        p1, p2 = np.array(e1[:, :, :, tf.newaxis]/d1), np.array(e2[:, :, :, tf.newaxis]/d1)
        p1[:, :, :, 0::2], p1[:, :, :, 1::2] = np.sin(p1[:, :, :, 0::2]), np.cos(p1[:, :, :, 1::2])
        p2[:, :, :, 0::2], p2[:, :, :, 1::2] = np.sin(p2[:, :, :, 0::2]), np.cos(p2[:, :, :, 1::2])
        return tf.concat([p1, p2], -1)


class Attention(keras.layers.Layer):
    def __init__(self, ln, an, head, size, attdrop=0., drop=0., cross=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.head, self.size, self.dm, self.ninf = head, size, head*size, -1e6
        self.aw = self.add_weight(ln+an+'in_proj_weight', [self.dm, self.dm*3], None, 'random_normal', trainable=True)
        self.ab = self.add_weight(ln+an+'in_proj_bias', [self.dm*3, ], None, 'random_normal', trainable=True)
        self.dense = keras.layers.Dense(self.dm, None, True, name=ln+an+'out_proj')
        self.norm = keras.layers.LayerNormalization(-1, 1e-6, name=ln+('norm1' if not cross else 'norm2'))
        self.attdrop = keras.layers.Dropout(attdrop)
        self.drop = keras.layers.Dropout(drop)

    def transposing(self, x):
        return tf.transpose(tf.reshape(x, [-1, tf.shape(x)[1], self.head, self.size]), [0, 2, 1, 3])

    def calculating(self, q, k, v, mask, training):
        a1 = tf.matmul(self.transposing(q), self.transposing(k), transpose_b=True)
        a1 = a1/tf.math.sqrt(tf.cast(self.size, tf.float32))
        m1 = None if mask is None else tf.cast(mask[:, tf.newaxis, tf.newaxis, :], tf.float32)*self.ninf
        a1 = tf.nn.softmax(a1 if mask is None else a1+m1, axis=-1)
        return tf.matmul(self.attdrop(a1, training=training), self.transposing(v)), a1

    def propagating(self, q, k, v, x=None, mask=None, training=False):
        w1, b1 = tf.split(self.aw, 3, 1), tf.split(self.ab, 3, 0)
        q1, k1, v1 = tf.matmul(q, w1[0])+b1[0], tf.matmul(k, w1[1])+b1[1], tf.matmul(v, w1[2])+b1[2]
        x1, a1 = self.calculating(q1, k1, v1, mask, training)
        x1 = tf.reshape(tf.transpose(x1, [0, 2, 1, 3]), [-1, tf.shape(x1)[2], self.dm])
        return self.norm((v if x is None else x)+self.drop(self.dense(x1), training=training)), a1


class SimpleAttention(keras.layers.Layer):
    def __init__(self, ln, nq, dim, head, drop, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)
        self.nq, self.head, self.size, self.ninf = nq, head, dim//head, -1e6
        self.wq = keras.layers.Dense(dim, name=ln+'q_linear')
        self.wk = keras.layers.Dense(dim, name=ln+'k_linear')
        self.attdrop = keras.layers.Dropout(drop)

    def propagating(self, q, k, mask, training):
        q1, k1, s1 = self.wq(q), self.wk(k), tf.shape(k)
        q1 = tf.reshape(q1, [-1, tf.shape(q1)[1], self.head, self.size])
        k1 = tf.transpose(tf.reshape(k1, [-1, tf.shape(k1)[1], self.head, self.size]), [0, 2, 3, 1])
        a1 = tf.einsum('bqnc,bncl->bqnl', q1, k1)/tf.math.sqrt(tf.cast(self.size, tf.float32))
        a1 = a1 if mask is None else a1+tf.cast(mask[:, tf.newaxis, tf.newaxis, :], tf.float32)*self.ninf
        a1 = tf.nn.softmax(tf.reshape(a1, [-1, self.nq, self.head*s1[1]]), axis=-1)
        return self.attdrop(a1, training=training)


class TransEncoder(keras.layers.Layer):
    def __init__(self, ln, head, size, dff, drop=0., act='relu', **kwargs):
        super(TransEncoder, self).__init__(**kwargs)
        self.att = Attention(ln, 'self_attn.', head, size, drop, drop)
        self.norm = keras.layers.LayerNormalization(-1, 1e-6, name=ln+'norm2')
        self.dense1 = keras.layers.Dense(dff, act, True, name=ln+'linear1')
        self.dense2 = keras.layers.Dense(int(head*size), None, True, name=ln+'linear2')
        self.drop1 = keras.layers.Dropout(drop)
        self.drop2 = keras.layers.Dropout(drop)

    def propagating(self, x, pos, mask=None, training=False):
        x1, a1 = self.att.propagating(x+pos, x+pos, x, None, mask, training)
        x2 = self.dense2(self.drop1(self.dense1(x1), training=training))
        return self.norm(x1+self.drop2(x2, training=training))


class TransDecoder(keras.layers.Layer):
    def __init__(self, ln, head, size, dff, drop=0., act='relu', **kwargs):
        super(TransDecoder, self).__init__(**kwargs)
        self.att1 = Attention(ln, 'self_attn.', head, size, drop, drop)
        self.att2 = Attention(ln, 'multihead_attn.', head, size, drop, drop, True)
        self.norm = keras.layers.LayerNormalization(-1, 1e-6, name=ln+'norm3')
        self.dense1 = keras.layers.Dense(dff, act, True, name=ln+'linear1')
        self.dense2 = keras.layers.Dense(int(head*size), None, True, name=ln+'linear2')
        self.drop1 = keras.layers.Dropout(drop)
        self.drop2 = keras.layers.Dropout(drop)

    def propagating(self, x, memory, pos, emb, memmask=None, training=False):
        x1, a1 = self.att1.propagating(x+emb, x+emb, x, None, None, training)
        x2, a2 = self.att2.propagating(x1+emb, memory+pos, memory, x1, memmask, training)
        x3 = self.dense2(self.drop1(self.dense1(x2), training=training))
        return self.norm(x2+self.drop2(x3, training=training))


class Transformer(keras.layers.Layer):
    def __init__(self, ln, nq, enc=6, dec=6, head=8, size=32, dff=2048, drop=0., act='relu', **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.enc, self.dec, n1, n2 = enc, dec, 'transformer.encoder.layers.{}.', 'transformer.decoder.layers.{}.'
        self.encoder = [TransEncoder(ln+n1.format(i1), head, size, dff, drop, act) for i1 in range(enc)]
        self.decoder = [TransDecoder(ln+n2.format(i1), head, size, dff, drop, act) for i1 in range(dec)]
        self.emb = self.add_weight(ln+'query_embed.weight', (nq, head*size), None)
        self.norm = keras.layers.LayerNormalization(-1, 1e-6, name=ln+'transformer.decoder.norm')

    def propagating(self, image, pos, mask, training=False):
        x1, t1, e1, r1 = None, None, tf.tile(tf.expand_dims(self.emb, 0), [tf.shape(image)[0], 1, 1]), []

        for i1 in range(self.enc):
            x1 = self.encoder[i1].propagating(image if x1 is None else x1, pos, mask, training)

        for i1 in range(self.dec):
            t1 = self.decoder[i1].propagating(tf.zeros_like(e1) if t1 is None else t1, x1, pos, e1, mask, training)
            r1.append(self.norm(t1))

        return self.norm(t1), tf.stack(r1), x1


class MaskHead(keras.layers.Layer):
    def __init__(self, ln, dim, **kwargs):
        super(MaskHead, self).__init__(**kwargs)
        self.outpad = keras.layers.ZeroPadding2D(1)
        self.outconv = keras.layers.Conv2D(1, 3, 1, 'VALID', name=ln+'out_lay')
        self.pad = [keras.layers.ZeroPadding2D(1) for _ in range(len(dim))]
        self.conv = [keras.layers.Conv2D(dim[i1], 3, 1, 'VALID', name=ln+'lay'+str(i1+1)) for i1 in range(len(dim))]
        self.norm = [tfa.layers.GroupNormalization(8, -1, 1e-5, name=ln+'gn'+str(i1+1)) for i1 in range(len(dim))]
        self.adap = [keras.layers.Conv2D(dim[i1+1], 1, name=ln+'adapter'+str(i1+1)) for i1 in range(len(dim)-2)]

    def propagating(self, proj, mask, feature):
        s1, s2, s3 = tf.shape(proj), tf.shape(mask), tf.shape(feature[-1])
        x1 = tf.concat([tf.repeat(proj, s2[1], 0), tf.reshape(mask, [-1, s2[2], s2[3], s2[4]])], -1)
        x1 = tf.nn.relu(self.norm[0](self.conv[0](self.pad[0](x1))))
        x1 = tf.nn.relu(self.norm[1](self.conv[1](self.pad[1](x1))))

        for i1 in range(len(self.adap)):
            f1 = tf.repeat(self.adap[i1](feature[i1]), s2[1], 0)
            x1 = f1+tf.image.resize(x1, tf.shape(f1)[1:3], 'nearest')
            x1 = tf.nn.relu(self.norm[i1+2](self.conv[i1+2](self.pad[i1+2](x1))))

        return tf.reshape(tf.squeeze(self.outconv(self.outpad(x1)), -1), [s1[0], s2[1], s3[1], s3[2]])


class DETR(keras.layers.Layer):
    def __init__(self, small=True, nq=100, layer=6, dim=256, head=8, dff=2048, cls=250, mlp=3, drop=0., **kwargs):
        super(DETR, self).__init__(**kwargs)
        self.nq, self.head, self.size = nq, head, dim//head
        self.rpl = {'/': '.', 'kernel': 'weight', 'gamma': 'weight', 'beta': 'bias', 'mov': 'runn', 'variance': 'var'}
        self.posemb = PosEmbedding(dim//2)
        self.backbone = ResNet('detr.backbone.0.body.', small)
        self.proj = keras.layers.Conv2D(dim, 1, 1, name='detr.input_proj')
        self.trans = Transformer('detr.', nq, layer, layer, head, dim//head, dff, drop, 'relu')
        self.cls = keras.layers.Dense(cls+1, name='detr.class_embed')
        self.att = SimpleAttention('bbox_attention.', nq, dim, head, drop)
        self.mh = MaskHead('mask_head.', [dim+head, dim//2, dim//4, dim//8, dim//16])
        self.mlp = [keras.layers.Dense(
            dim if i1 != mlp-1 else 4, name='detr.bbox_embed.layers.{}'.format(i1)) for i1 in range(mlp)]

    def loading(self, pth):
        s1, l1 = self.propagating(tf.ones((2, 224, 224, 3))), torch.load(pth)['model']
        r1, t1 = re.compile('|'.join(map(re.escape, self.rpl))), []
        n1 = [r1.sub((lambda x1: self.rpl[x1.group(0)]), i1.name[:-2]) for i1 in self.weights]

        for i1 in n1:
            if len(l1[i1].shape) == 4:
                t1.append(tf.transpose(l1[i1].numpy(), [2, 3, 1, 0]))
            elif 'weight' in i1 and 'query_embed' not in i1 and len(l1[i1].shape) == 2:
                t1.append(tf.transpose(l1[i1].numpy(), [1, 0]))
            else:
                t1.append(l1[i1].numpy())

        keras.backend.batch_set_value(zip(self.weights, t1))

    def propagating(self, x, training=False):
        l1 = self.backbone.propagating(x, training=training)
        p1 = self.posemb.propagating(tf.zeros_like(l1[-1][:, :, :, 0]))
        x1, s1 = self.proj(l1[-1]), tf.shape(p1)
        x2, p1 = tf.reshape(x1, [s1[0], s1[1]*s1[2], s1[3]]), tf.reshape(p1, [s1[0], s1[1]*s1[2], s1[3]])
        x2, r1, m1 = self.trans.propagating(x2, p1, None)
        c1, c2, a1 = self.cls(r1), r1, self.att.propagating(r1[-1], m1, None, training)

        for i1 in range(len(self.mlp)):
            c2 = tf.nn.relu(self.mlp[i1](c2)) if i1 != len(self.mlp)-1 else tf.nn.sigmoid(self.mlp[i1](c2))

        a1 = tf.transpose(tf.reshape(a1, [-1, self.nq, self.head, s1[1], s1[2]]), [0, 1, 3, 4, 2])
        return c1[-1], c2[-1], self.mh.propagating(x1, a1, l1[:3][::-1])
