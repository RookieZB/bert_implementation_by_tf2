# -*- coding: utf-8 -*-

"""
Simple TextCNN Implementation for Text Classification

"""

import tensorflow as tf
import tensorflow.keras as keras


class TextCNN(keras.Model):
    def __init__(self, vocsize, embsize, kernel, outdim, cate, drop, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.config = {'kernel': kernel}
        self.emb = keras.layers.Embedding(vocsize, embsize)
        self.conv = [[keras.layers.Conv1D(j1[1], j1[0], padding='same') for j1 in i1] for i1 in kernel]
        self.pool = keras.layers.GlobalMaxPool1D()
        self.dense = keras.layers.Dense(outdim, activation='relu')
        self.drop = keras.layers.Dropout(drop)
        self.cls = keras.layers.Dense(cate)

    def call(self, x, softmax=False, **kwargs):
        x1 = self.emb(x)

        for i1 in self.conv:
            x2 = [j1(x1) for j1 in i1]
            x1 = x2[0] if len(x2) == 1 else tf.concat(x2, -1)

        x1 = self.cls(self.drop(self.dense(self.pool(x1))))
        return tf.nn.softmax(x1) if softmax else x1

    def get_config(self):
        return self.config


def textcnn_training():  # Test accuracy is about 0.86.
    import tensorflow_datasets as tfds

    lr_1, epoch_1, batch_1, buffer_1 = 5e-4, 5, 64, 10000
    kernel_1, embsize_1, outdim_1, cate_1, drop_1 = [[[2, 32], [3, 32], [4, 32]], [[3, 32]]], 64, 64, 1, 0.5
    data_1, info_1 = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
    train_1 = data_1['train'].shuffle(buffer_1).padded_batch(batch_1)
    test_1 = data_1['test'].padded_batch(batch_1)
    model_1 = TextCNN(info_1.features['text'].encoder.vocab_size, embsize_1, kernel_1, outdim_1, cate_1, drop_1)
    loss_1 = keras.losses.BinaryCrossentropy(from_logits=True)
    model_1.compile(keras.optimizers.Adam(lr_1), loss_1, ['accuracy'])
    model_1.fit(train_1, epochs=epoch_1, validation_data=test_1, validation_steps=30)
    return model_1
