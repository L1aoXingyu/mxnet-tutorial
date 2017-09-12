import mxnet as mx
import mxnet.gluon as g
import numpy as np

ctx = mx.gpu()
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

trigram = [((test_sentence[i], test_sentence[i + 1]), test_sentence[i + 2])
           for i in range(len(test_sentence) - 2)]

vocb = set(test_sentence)
word_to_idx = {word: i for i, word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}


class NgramModel(g.Block):
    def __init__(self, vocb_size, context_size, n_dim):
        super(NgramModel, self).__init__()
        with self.name_scope():
            self.n_word = vocb_size
            self.embedding = g.nn.Embedding(self.n_word, n_dim)
            self.linear1 = g.nn.Dense(128, activation='relu')
            self.linear2 = g.nn.Dense(self.n_word)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.reshape((1, -1))
        out = self.linear1(emb)
        out = self.linear2(out)
        log_prob = mx.nd.log_softmax(out)
        return log_prob


ngrammodel = NgramModel(len(word_to_idx), CONTEXT_SIZE, 100)
ngrammodel.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
criterion = g.loss.SoftmaxCrossEntropyLoss()
optimizer = g.Trainer(ngrammodel.collect_params(), 'adam',
                      {'learning_rate': 1e-3})

for epoch in range(100):
    print('epoch: {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0
    for data in trigram:
        word, label = data
        word = mx.nd.array(
            [word_to_idx[i] for i in word], ctx=ctx, dtype=np.int32)
        label = mx.nd.array([word_to_idx[label]], ctx=ctx, dtype=np.int32)
        # forward
        with mx.autograd.record():
            out = ngrammodel(word)
            loss = criterion(out, label)
        running_loss += mx.nd.mean(loss).asscalar()
        # backward
        loss.backward()
        optimizer.step(1)
    print('Loss: {:.6f}'.format(running_loss / len(word_to_idx)))

word, label = trigram[3]
word = mx.nd.array([word_to_idx[i] for i in word], ctx=ctx, dtype=np.int32)
out = ngrammodel(word)
predict_label = mx.nd.argmax(out, axis=1)
predict_word = idx_to_word[predict_label.asscalar().astype(np.int32)]
print('real word is {}, predict word is {}'.format(label, predict_word))
