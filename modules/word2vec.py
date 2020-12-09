# -*- coding: utf-8 -*-

"""
Simple Word2Vec Implementation
https://tensorflow.google.cn/tutorials/text/word2vec
https://github.com/RaRe-Technologies/gensim

"""

import tensorflow as tf
import tensorflow.keras as keras
from annoy import AnnoyIndex


class Word2Vec(keras.Model):
    def __init__(self, vocab, dim, tree=100, **kwargs):
        super(Word2Vec, self).__init__(**kwargs)
        self.vocab, self.vocsize, self.dim, self.tree = vocab, len(vocab), dim, tree
        self.taremb = keras.layers.Embedding(self.vocsize, dim, name='w2v')
        self.conemb = keras.layers.Embedding(self.vocsize, dim)
        self.dot = keras.layers.Dot((3, 2))
        self.flat = keras.layers.Flatten()
        self.annoy, self.normemb = None, None

    def call(self, pair, softmax=False, **kwargs):
        e1 = [self.conemb(pair[1]), self.taremb(pair[0])]
        x1 = self.flat(self.dot(e1))
        return tf.nn.softmax(x1) if softmax else x1

    def update(self):
        v1 = self.taremb.weights[0]
        self.normemb = v1/tf.norm(v1, 2, -1, True)
        self.annoy = AnnoyIndex(self.dim, metric='angular')

        for i1, v2 in enumerate(self.normemb):
            self.annoy.add_item(i1, v2)

        self.annoy.build(self.tree)

    def search(self, key, num=5, annoy=False):
        k1 = self.normemb[self.vocab.index(key)]

        if annoy:
            l1, d1 = self.annoy.get_nns_by_vector(k1, num, include_distances=True)
            return [self.vocab[i1] for i1 in l1]
        else:
            d1 = tf.matmul(self.normemb, tf.expand_dims(k1, -1))[:, 0]
            l1 = tf.argsort(d1, direction='DESCENDING').numpy().tolist()
            return [self.vocab[i1] for i1 in l1][:num]


def w2v_training():  # Final loss is about 0.47.
    import re
    import string

    def data_generating(data, window, ns, vocab):
        tar1, con1, label1 = [], [], []
        tab1 = keras.preprocessing.sequence.make_sampling_table(vocab)

        for seq1 in data:
            sg1, _ = keras.preprocessing.sequence.skipgrams(seq1, vocab, window, 0, sampling_table=tab1)

            for tar2, con2 in sg1:
                true1 = tf.expand_dims(tf.constant([con2], 'int64'), 1)
                samp1, _, _ = tf.random.log_uniform_candidate_sampler(true1, 1, ns, True, vocab)
                con1.append(tf.concat([true1, tf.expand_dims(samp1, 1)], 0))
                label1.append(tf.constant([1]+[0]*ns, 'int64'))
                tar1.append(tar2)

        return tar1, con1, label1

    def text_standardizing(data):
        return tf.strings.regex_replace(
            tf.strings.lower(data), '[%s]' % re.escape(string.punctuation), '')

    maxvoc_1, maxlen_1, window_1, ns_1 = 4096, 10, 2, 4
    epoch_1, batch_1, buffer_1, dim_1 = 20, 1024, 10000, 128
    site_1 = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
    path_1 = keras.utils.get_file('shakespeare.txt', site_1)
    file_1 = tf.data.TextLineDataset(path_1)
    file_1 = file_1.filter(lambda x: tf.cast(tf.strings.length(x), bool))

    pre_1 = keras.layers.experimental.preprocessing.TextVectorization
    vect_1 = pre_1(maxvoc_1, text_standardizing, output_mode='int', output_sequence_length=maxlen_1)
    vect_1.adapt(file_1.batch(batch_1))

    file_1 = file_1.batch(batch_1).prefetch(tf.data.experimental.AUTOTUNE)
    list_1 = list(file_1.map(vect_1).unbatch().as_numpy_iterator())
    tar_1, con_1, label_1 = data_generating(list_1, window_1, ns_1, maxvoc_1)

    dataset_1 = tf.data.Dataset.from_tensor_slices(((tar_1, con_1), label_1))
    dataset_1 = dataset_1.shuffle(buffer_1).batch(batch_1, drop_remainder=True)
    dataset_1 = dataset_1.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model_1 = Word2Vec(vect_1.get_vocabulary(), dim_1)
    model_1.compile('adam', keras.losses.CategoricalCrossentropy(from_logits=True), ['accuracy'])
    model_1.fit(dataset_1, epochs=epoch_1)
    model_1.update()
    return model_1
