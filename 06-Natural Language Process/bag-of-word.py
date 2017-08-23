__author__ = 'SherlockLiao'

import mxnet as mx
import mxnet.gluon as g
import numpy as np

ctx = mx.gpu()
# The Continuous Bag-of-Words model (CBOW) is frequently used in NLP deep learning. It is a model that tries to predict words given the context of a few words before and a few words after the target word. This is distinct from language modeling, since CBOW is not sequential and does not have to be probabilistic. Typcially, CBOW is used to quickly train word embeddings, and these embeddings are used to initialize the embeddings of some more complicated model. Usually, this is referred to as pretraining embeddings. It almost always helps performance a couple of percent.

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
word_to_idx = {word: i for i, word in enumerate(vocab)}

data = []
for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
    context = [
        raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]
    ]
    target = raw_text[i]
    data.append((context, target))


class CBOW(g.Block):
    def __init__(self, n_word, n_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = g.nn.Embedding(n_word, n_dim)
        self.linear1 = g.nn.Dense(128, activation='relu')
        self.linear2 = g.nn.Dense(n_word)

    def forward(self, x):
        x = self.embedding(x)
        x = x.reshape((1, -1))
        x = self.linear1(x)
        x = self.linear2(x)
        x = mx.nd.log_softmax(x)
        return x


model = CBOW(len(word_to_idx), 100, CONTEXT_SIZE)
model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

criterion = g.loss.SoftmaxCrossEntropyLoss()
optimizer = g.Trainer(model.collect_params(), 'sgd', {'learning_rate': 1e-2})

for epoch in range(100):
    print('epoch {}'.format(epoch))
    print('*' * 10)
    running_loss = 0
    for word in data:
        context, target = word
        context = mx.nd.array(
            [word_to_idx[i] for i in context], ctx=ctx).astype(np.int32)
        # context = context.astype(np.int32)
        target = mx.nd.array([word_to_idx[target]], ctx=ctx).astype(np.int32)
        # target = target.astype(np.int32)
        # forward
        with g.autograd.record():
            out = model(context)
            loss = criterion(out, target)
        running_loss += mx.nd.mean(loss).asscalar()
        # backward
        loss.backward()
        optimizer.step(1)
    print('loss: {:.6f}'.format(running_loss / len(data)))
