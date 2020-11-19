# -*- coding: utf-8 -*-

"""
ConvBERT Implementation (not sure if is correct and inference speed is still a problem to fix)
https://github.com/yitu-opensource/ConvBert

"""

import re
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import mymodels as mm


class GroupDense(keras.layers.Layer):  # GroupDense operation not finished.
    def __init__(self, **kwargs):
        super(GroupDense, self).__init__(**kwargs)


class ConvAttention(keras.layers.Layer):
    def __init__(self, bname, lname, head, size, ratio, kernel, attdrop, drop, ninf=-1e4, **kwargs):
        super(ConvAttention, self).__init__(**kwargs)
        self.head, self.size, self.dim, self.kernel, self.ninf = head//ratio, size, head*size//ratio, kernel, ninf
        self.wq = keras.layers.Dense(self.dim, None, True, mm.w_initializing(), name=bname+lname[0])
        self.wk = keras.layers.Dense(self.dim, None, True, mm.w_initializing(), name=bname+lname[1])
        self.wv = keras.layers.Dense(self.dim, None, True, mm.w_initializing(), name=bname+lname[2])
        self.dense = keras.layers.Dense(head*size, None, True, mm.w_initializing(), name=bname+lname[3])
        self.cc = keras.layers.SeparableConv1D(self.dim, kernel, padding='same', name=bname+lname[8])
        self.ck = keras.layers.Dense(self.head*kernel, None, True, mm.w_initializing(), name=bname+lname[9])
        self.co = keras.layers.Dense(self.dim, None, True, mm.w_initializing(), name=bname+lname[10])
        self.norm = keras.layers.LayerNormalization(-1, 1e-6, name=bname+lname[4])
        self.attdrop = keras.layers.Dropout(attdrop)
        self.drop = keras.layers.Dropout(drop)

    def transposing(self, x):
        return tf.transpose(tf.reshape(x, [-1, tf.shape(x)[1], self.head, self.size]), [0, 2, 1, 3])

    def masking(self, mask):
        return tf.cast(mask[:, tf.newaxis, tf.newaxis, :], tf.float32)*self.ninf

    def convoluting(self, x, q):
        c1, c2 = tf.reshape(self.ck(q*self.cc(x)), [-1, self.kernel, 1]), self.co(x)
        c2 = tf.image.extract_patches(tf.expand_dims(c2, -1), [1, self.kernel, 1, 1], [1]*4, [1]*4, 'SAME')
        return tf.matmul(tf.reshape(c2, [-1, self.size, self.kernel]), tf.nn.softmax(c1, axis=1))

    def calculating(self, x, mask, training):
        q1, k1, v1, l1 = self.wq(x), self.wk(x), self.wv(x), tf.shape(x)[1]
        c1 = tf.reshape(self.convoluting(x, q1), [-1, l1, self.head, self.size])
        a1 = tf.matmul(self.transposing(q1), self.transposing(k1), transpose_b=True)
        a1 = a1/tf.math.sqrt(tf.cast(self.size, tf.float32))
        a1 = tf.nn.softmax(a1+self.masking(mask) if mask is not None else a1, axis=-1)
        a2, v1 = self.attdrop(a1, training=training), self.transposing(v1)
        a2 = tf.transpose(tf.matmul(a2, v1), [0, 2, 1, 3])
        return tf.reshape(tf.concat([a2, c1], 2), [-1, l1, self.dim*2]), a1

    def propagating(self, x, mask=None, training=False):
        x1, a1 = self.calculating(x, mask, training)
        return self.norm(x+self.drop(self.dense(x1), training=training)), a1


class ConvTransEncoder(keras.layers.Layer):
    def __init__(self, bname, lname, head, size, ratio, kernel, dff, attdrop, drop, act, **kwargs):
        super(ConvTransEncoder, self).__init__(**kwargs)
        self.att = ConvAttention(bname, lname, head, size, ratio, kernel, attdrop, drop)
        self.norm = keras.layers.LayerNormalization(-1, 1e-6, name=bname+lname[7])
        self.dense1 = keras.layers.Dense(dff, act, True, mm.w_initializing(), name=bname+lname[5])
        self.dense2 = keras.layers.Dense(int(head*size), None, True, mm.w_initializing(), name=bname+lname[6])
        self.drop = keras.layers.Dropout(drop)

    def propagating(self, x, mask=None, training=False):
        x1, a1 = self.att.propagating(x, mask, training)
        return self.norm(x1+self.drop(self.dense2(self.dense1(x1)), training=training))


class ConvBERT(keras.layers.Layer):
    def __init__(self, size, **kwargs):
        super(ConvBERT, self).__init__(**kwargs)
        self.param, self.hd = self.checking(size), 'electra'
        self.act = mm.gelu_activating if self.param['hidden_act'] == 'gelu' else self.param['hidden_act']
        self.replacement = {'embeddings/embeddings': 'embeddings'}
        self.namee = ['/word_embeddings', '/position_embeddings', '/token_type_embeddings', '/LayerNorm']
        self.namea = [
            '/attention/self/query',
            '/attention/self/key',
            '/attention/self/value',
            '/attention/output/dense',
            '/attention/output/LayerNorm',
            '/intermediate/dense',
            '/output/dense',
            '/output/LayerNorm',
            '/attention/self/conv_attn_key',
            '/attention/self/conv_attn_kernel',
            '/attention/self/conv_attn_point']
        self.embedding = mm.Embedding(
            self.hd+'/embeddings',
            self.namee,
            self.param['vocab_size'],
            self.param['embedding_size'],
            self.param['max_position_embeddings'],
            self.param['type_vocab_size'],
            float(self.param['hidden_dropout_prob']))
        self.projection = keras.layers.Dense(
            self.param['hidden_size'],
            kernel_initializer=mm.w_initializing(),
            name=self.hd+'/embeddings_project') if self.param['embedding_size'] != self.param['hidden_size'] else None
        self.encoder = [ConvTransEncoder(
            self.hd+'/encoder/layer_'+str(i1),
            self.namea,
            self.param['num_attention_heads'],
            self.param['hidden_size']//self.param['num_attention_heads'],
            self.param['head_ratio'],
            self.param['conv_kernel_size'],
            self.param['intermediate_size'],
            float(self.param['attention_probs_dropout_prob']),
            float(self.param['hidden_dropout_prob']),
            self.act) for i1 in range(self.param['num_hidden_layers'])]

    @staticmethod
    def checking(size):
        conf1 = {
            'small': {
                'vocab_size': 30522,
                'embedding_size': 128,
                'max_position_embeddings': 512,
                'type_vocab_size': 2,
                'hidden_size': 256,
                'num_attention_heads': 4,
                'intermediate_size': 1024,
                'num_hidden_layers': 12,
                'conv_kernel_size': 9,
                'head_ratio': 2,
                'hidden_dropout_prob': 0.1,
                'attention_probs_dropout_prob': 0.1,
                'hidden_act': 'gelu'},
            'base': {
                'vocab_size': 30522,
                'embedding_size': 768,
                'max_position_embeddings': 512,
                'type_vocab_size': 2,
                'hidden_size': 768,
                'num_attention_heads': 12,
                'intermediate_size': 3072,
                'num_hidden_layers': 12,
                'conv_kernel_size': 9,
                'head_ratio': 2,
                'hidden_dropout_prob': 0.1,
                'attention_probs_dropout_prob': 0.1,
                'hidden_act': 'gelu'}}
        return conf1[size]

    def loading(self, ckpt):
        _ = self.propagating(tf.ones((2, 2), tf.int32), tf.zeros((2, 2)), tf.zeros((2, 2)))
        r1 = re.compile('|'.join(map(re.escape, self.replacement)))
        n1 = [r1.sub((lambda x1: self.replacement[x1.group(0)]), i1.name[:-2]) for i1 in self.weights]
        keras.backend.batch_set_value(zip(self.weights, [tf.train.load_variable(ckpt, i1) for i1 in n1]))

    def propagating(self, x, seg, mask, training=False):
        x1 = self.embedding.propagating(x, seg, training=training)
        x1 = self.projection(x1) if self.projection else x1

        for i1 in range(self.param['num_hidden_layers']):
            x1 = self.encoder[i1].propagating(x1, mask, training=training)

        return x1


def convbert_testing(sentence=None):
    toke1 = mm.Tokenizer()
    toke1.loading('../models/convbert_small_en/vocab.txt')
    bert1 = ConvBERT('small')
    bert1.loading('../models/convbert_small_en/model.ckpt')
    text1, segm1, mask1 = toke1.encoding('Have a good day.' if not sentence else sentence)
    print(bert1.propagating(np.array([text1]), np.array([segm1]), np.array([mask1])))


def convbert_finetuning():  # Dev accuracy is about 0.9037.
    import time
    import nlp
    import pandas as pd

    m_1, vocab_1, ckpt_1 = 'small', './models/convbert_small_en/vocab.txt', './models/convbert_small_en/model.ckpt'
    lr_1, drop_1, cate_1, maxlen_1, batch_1, epoch_1 = 1e-4, 0.5, 2, 64, 32, 5
    key_1 = ['embedding']+['encoder/layer_'+str(i_1) for i_1 in range(12)]
    ldecay_1 = dict(zip(key_1, [0.8**(len(key_1)-i_1) for i_1 in range(len(key_1))]))

    def data_processing(data, tokenizer, maxlen, batch, training):
        text1, seg1, mask1, label1 = [], [], [], []

        for i in range(len(data)):
            text2, seg2, mask2 = tokenizer.encoding(data['sentence'][i], None, maxlen)
            text1, seg1, mask1 = text1+[text2], seg1+[seg2], mask1+[mask2]
            label1.append(data['label'][i])

        text1, seg1, mask1, label1 = np.array(text1), np.array(seg1), np.array(mask1), np.array(label1)
        data1 = tf.data.Dataset.from_tensor_slices((text1, seg1, mask1, label1))
        return data1.shuffle(len(text1)).batch(batch) if training else data1.batch(batch)

    class ModelBERT(keras.Model):
        def __init__(self, model, drop, category):
            super(ModelBERT, self).__init__()
            self.bert = ConvBERT(model)
            self.drop = keras.layers.Dropout(drop)
            self.dense = keras.layers.Dense(category, activation='softmax')

        def propagating(self, text, segment, mask, training):
            x1 = self.bert.propagating(text, segment, mask, training)[:, 0, :]
            return self.dense(self.drop(x1, training=training))

    @tf.function
    def step_training(text, segment, mask, y):
        with tf.GradientTape() as tape_1:
            pred_1 = model_1.propagating(text, segment, mask, True)
            value_1 = function_1(y, pred_1)

        grad_1 = tape_1.gradient(value_1, model_1.trainable_variables)
        grad_1, _ = tf.clip_by_global_norm(grad_1, 1.0)
        optimizer_1.apply_gradients(zip(grad_1, model_1.trainable_variables))
        loss_1(value_1)
        acc_1(y, pred_1)

    @tf.function
    def step_evaluating(text, segment, mask, y):
        pred_1 = model_1.propagating(text, segment, mask, False)
        acc_2(y, pred_1)

    tokenizer_1 = mm.Tokenizer()
    tokenizer_1.loading(vocab_1)
    data_1 = nlp.load_dataset('glue', 'sst2')
    training_1, dev_1 = pd.DataFrame(data_1['train']), pd.DataFrame(data_1['validation'])
    training_2 = data_processing(training_1, tokenizer_1, batch_1, maxlen_1, True)
    dev_2 = data_processing(dev_1, tokenizer_1, batch_1, maxlen_1, False)

    model_1 = ModelBERT(m_1, drop_1, cate_1)
    model_1.bert.loading(ckpt_1)
    function_1 = keras.losses.SparseCategoricalCrossentropy()
    optimizer_1 = mm.AdamW(epoch_1*(int(len(training_1)/batch_1)+1), lr_1, lmode=2, ldecay=ldecay_1)

    loss_1 = tf.keras.metrics.Mean(name='training_loss')
    acc_1 = tf.keras.metrics.SparseCategoricalAccuracy(name='training_accuracy')
    acc_2 = tf.keras.metrics.SparseCategoricalAccuracy(name='dev_accuracy')
    temp_1 = 'Training loss is {:.4f}, and accuracy is {:.4f}.'
    temp_2 = 'Dev accuracy is {:.4f}, and epoch cost is {:.4f}.'

    for e_1 in range(epoch_1):
        print('Epoch {} running.'.format(e_1+1))
        count_1, time_0 = 0, time.time()

        for x_1, x_2, x_3, y_1 in training_2:
            time_1, count_1 = time.time(), count_1+1
            step_training(x_1, x_2, x_3, y_1)

            if count_1 % 250 == 0:
                print(temp_1.format(float(loss_1.result()), float(acc_1.result())))

        for x_1, x_2, x_3, y_1 in dev_2:
            step_evaluating(x_1, x_2, x_3, y_1)

        print(temp_2.format(float(acc_2.result()), time.time()-time_0))
        print('**********')
        acc_1.reset_states()
        acc_2.reset_states()

    return model_1
