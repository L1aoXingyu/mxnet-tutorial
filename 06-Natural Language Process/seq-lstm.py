__author__ = 'SherlockLiao'

import mxnet as mx
import mxnet.gluon as g
import numpy as np

ctx = mx.gpu()

training_data = [("The dog ate the apple".split(),
                  ["DET", "NN", "V", "DET", "NN"]),
                 ("Everybody read that book".split(), ["NN", "V", "DET",
                                                       "NN"])]

word_to_idx = {}
tag_to_idx = {}
for context, tag in training_data:
    for word in context:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
    for label in tag:
        if label not in tag_to_idx:
            tag_to_idx[label] = len(tag_to_idx)
alphabet = 'abcdefghijklmnopqrstuvwxyz'
character_to_idx = {}
for i in range(len(alphabet)):
    character_to_idx[alphabet[i]] = i


class CharLSTM(g.Block):
    def __init__(self, n_char, char_dim, char_hidden):
        super(CharLSTM, self).__init__()
        with self.name_scope():
            self.char_embedding = g.nn.Embedding(n_char, char_dim)
            self.char_lstm = g.rnn.LSTM(char_hidden)

    def forward(self, x, hidden):
        x = self.char_embedding(x)
        x = mx.nd.transpose(x, (1, 0, 2))
        _, h = self.char_lstm(x, hidden)
        return h[0]


class LSTMTagger(g.Block):
    def __init__(self, n_word, n_char, char_dim, n_dim, char_hidden, n_hidden,
                 n_tag):
        super(LSTMTagger, self).__init__()
        with self.name_scope():
            self.word_embedding = g.nn.Embedding(n_word, n_dim)
            self.char_lstm = CharLSTM(n_char, char_dim, char_hidden)
            self.lstm = g.rnn.LSTM(n_hidden)
            self.linear = g.nn.Dense(n_tag)

    def forward(self, x, char_hidden, hidden, word):
        for i, each in enumerate(word):
            char_list = []
            for letter in each:
                char_list.append(character_to_idx[letter.lower()])
            char_list = mx.nd.array(char_list, dtype=np.int32, ctx=ctx)
            char_list = mx.nd.expand_dims(char_list, axis=0)
            tempchar = self.char_lstm(char_list, char_hidden)
            t_shape = tempchar.shape
            tempchar = tempchar.reshape((t_shape[0], t_shape[2]))
            if i == 0:
                char = tempchar
            else:
                char = mx.nd.concat(char, tempchar, dim=0)
        x = self.word_embedding(x)
        x = mx.nd.concat(x, char, dim=1)
        x = mx.nd.expand_dims(x, axis=1)
        x, _ = self.lstm(x, hidden)
        x = x.reshape((x.shape[0], x.shape[2]))
        x = self.linear(x)
        y = mx.nd.log_softmax(x)
        return y


model = LSTMTagger(
    len(word_to_idx), len(character_to_idx), 10, 100, 50, 128, len(tag_to_idx))

model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

criterion = g.loss.SoftmaxCrossEntropyLoss()
optimizer = g.Trainer(model.collect_params(), 'sgd', {'learning_rate': 1e-2})


def make_sequence(x, dic):
    idx = mx.nd.array([dic[i] for i in x], ctx=ctx, dtype=np.int32)
    return idx


for epoch in range(300):
    print('*' * 10)
    print('epoch {}'.format(epoch + 1))
    running_loss = 0
    for data in training_data:
        word, tag = data
        word_list = make_sequence(word, word_to_idx)
        tag = make_sequence(tag, tag_to_idx)
        # forward
        h0 = mx.nd.zeros((1, 1, 128), ctx=ctx)
        c0 = mx.nd.zeros((1, 1, 128), ctx=ctx)
        char_h0 = mx.nd.zeros((1, 1, 50), ctx=ctx)
        char_c0 = mx.nd.zeros((1, 1, 50), ctx=ctx)
        with mx.autograd.record():
            out = model(word_list, [char_h0, char_c0], [h0, c0], word)
            loss = criterion(out, tag)
        running_loss += mx.nd.mean(loss).asscalar()
        # backward
        loss.backward()
        optimizer.step(1)
    print('Loss: {}'.format(running_loss / len(data)))
print()
input = make_sequence("Everybody ate the apple".split(), word_to_idx)

h0 = mx.nd.zeros((1, 1, 128), ctx=ctx)
c0 = mx.nd.zeros((1, 1, 128), ctx=ctx)
char_h0 = mx.nd.zeros((1, 1, 50), ctx=ctx)
char_c0 = mx.nd.zeros((1, 1, 50), ctx=ctx)
out = model(input, [char_h0, char_c0], [h0, c0],
            "Everybody ate the apple".split())
print(out)
