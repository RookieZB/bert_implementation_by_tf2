# -*- coding: utf-8 -*-

"""
Simple RNN Implementation for Text Classification
https://tensorflow.google.cn/tutorials/text/text_classification_rnn

"""

import tensorflow as tf
import tensorflow.keras as keras


class TextRNN(keras.Model):
    def __init__(self, vocsize, embsize, dim, outdim, cate, drop, **kwargs):
        super(TextRNN, self).__init__(**kwargs)
        self.config = {'cell': dim}
        self.emb = keras.layers.Embedding(vocsize, embsize)
        self.dense = keras.layers.Dense(outdim, activation='relu')
        self.drop = keras.layers.Dropout(drop)
        self.cls = keras.layers.Dense(cate)
        self.rnn = [keras.layers.Bidirectional(keras.layers.LSTM(
            d1, return_sequences=(i1 < len(dim)-1))) for i1, d1 in enumerate(dim)]

    def call(self, x, softmax=False, **kwargs):
        x1 = self.emb(x)

        for i1 in range(len(self.rnn)):
            x1 = self.rnn[i1](x1)

        x1 = self.cls(self.drop(self.dense(x1)))
        return tf.nn.softmax(x1) if softmax else x1

    def get_config(self):
        return self.config


def textrnn_training():  # Test accuracy is about 0.86.
    import tensorflow_datasets as tfds

    lr_1, epoch_1, batch_1, buffer_1 = 1e-4, 5, 64, 10000
    embsize_1, dim_1, outdim_1, cate_1, drop_1 = 64, [64, 32], 64, 1, 0.5
    data_1, info_1 = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
    train_1 = data_1['train'].shuffle(buffer_1).padded_batch(batch_1)
    test_1 = data_1['test'].padded_batch(batch_1)
    model_1 = TextRNN(info_1.features['text'].encoder.vocab_size, embsize_1, dim_1, outdim_1, cate_1, drop_1)
    loss_1 = keras.losses.BinaryCrossentropy(from_logits=True)
    model_1.compile(keras.optimizers.Adam(lr_1), loss_1, ['accuracy'])
    model_1.fit(train_1, epochs=epoch_1, validation_data=test_1, validation_steps=30)
    return model_1
