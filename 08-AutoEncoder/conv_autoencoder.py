__author__ = 'SherlockLiao'

import mxnet as mx
from mxnet import gluon as g
from mxnet import nd
import torch
from torchvision.utils import save_image
import os

ctx = mx.gpu()

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3


def img_transform(data, label):
    data = data.transpose((2, 0, 1))
    return (data.astype('float32') / 255 - 0.5) / 0.5, label.astype('float32')


dataset = g.data.vision.MNIST(transform=img_transform)
dataloader = g.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(g.nn.HybridBlock):
    def __init__(self):
        super(autoencoder, self).__init__()
        with self.name_scope():
            self.encoder = g.nn.HybridSequential('encoder_')
            with self.encoder.name_scope():
                # b, 16, 10, 10
                self.encoder.add(
                    g.nn.Conv2D(
                        16, 3, strides=3, padding=1, activation='relu'))
                self.encoder.add(g.nn.MaxPool2D(2, 2))  # b, 16, 5, 5
                self.encoder.add(
                    g.nn.Conv2D(8, 3, strides=2, padding=1,
                                activation='relu'))  # b, 8, 3, 3
                self.encoder.add(g.nn.MaxPool2D(2, 1))  # b, 8, 2, 2

            self.decoder = g.nn.HybridSequential('decoder_')
            with self.decoder.name_scope():
                self.decoder.add(
                    g.nn.Conv2DTranspose(16, 3, strides=2, activation='relu'))
                self.decoder.add(
                    g.nn.Conv2DTranspose(
                        8, 5, strides=3, padding=1, activation='relu'))
                self.decoder.add(
                    g.nn.Conv2DTranspose(
                        1, 2, strides=2, padding=1, activation='tanh'))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder()
model.hybridize()
model.collect_params().initialize(mx.init.Xavier('gaussian'), ctx=ctx)

criterion = g.loss.L2Loss()
optimizer = g.Trainer(model.collect_params(), 'adam',
                      {'learning_rate': learning_rate,
                       'wd': 1e-5})

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.as_in_context(ctx)
        batch = img.shape[0]
        # ===================forward=====================
        with mx.autograd.record():
            output = model(img)
            loss = criterion(output, img)
        # ===================backward====================
        loss.backward()
        optimizer.step(batch)
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, nd.mean(loss).asscalar()))
    if epoch % 10 == 0:
        pic = to_img(torch.FloatTensor(output.asnumpy()))
        save_image(pic, './dc_img/image_{}.png'.format(epoch))

model.save_params('./con_autoencoder.params')