__author__ = 'SherlockLiao'

import os

import mxnet as mx
import numpy as np
import torch
from mxnet import gluon as g
from mxnet import ndarray as nd
from torchvision.utils import save_image

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3
ctx = mx.gpu()


def img_transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')


dataset = g.data.vision.MNIST(transform=img_transform)
dataloader = g.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


class VAE(g.HybridBlock):
    def __init__(self):
        super(VAE, self).__init__()
        with self.name_scope():
            self.fc1 = g.nn.Dense(400)
            self.fc21 = g.nn.Dense(20)
            self.fc22 = g.nn.Dense(20)
            self.fc3 = g.nn.Dense(400)
            self.fc4 = g.nn.Dense(784)
        # self.fc1 = g.nn.Dense(
        #     400,
        #     in_units=784,
        #     weight_initializer=mx.init.Uniform(1. / np.sqrt(784)),
        #     bias_initializer=mx.init.Uniform(1. / np.sqrt(784)))
        # self.fc21 = g.nn.Dense(
        #     20,
        #     in_units=400,
        #     weight_initializer=mx.init.Uniform(1. / np.sqrt(400)),
        #     bias_initializer=mx.init.Uniform(1. / np.sqrt(400)))
        # self.fc22 = g.nn.Dense(
        #     20,
        #     in_units=400,
        #     weight_initializer=mx.init.Uniform(1. / np.sqrt(400)),
        #     bias_initializer=mx.init.Uniform(1. / np.sqrt(400)))
        # self.fc3 = g.nn.Dense(
        #     400,
        #     in_units=20,
        #     weight_initializer=mx.init.Uniform(1. / np.sqrt(20)),
        #     bias_initializer=mx.init.Uniform(1. / np.sqrt(20)))
        # self.fc4 = g.nn.Dense(
        #     784,
        #     in_units=400,
        #     weight_initializer=mx.init.Uniform(1. / np.sqrt(784)),
        #     bias_initializer=mx.init.Uniform(1. / np.sqrt(784)))

    def encode(self, x):
        h1 = nd.Activation(self.fc1(x), 'relu')
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        '''
        mu is a number and logvar is a ndarray
        '''
        std = nd.exp(0.5 * logvar)
        eps = nd.random_normal(
            loc=0, scale=1, shape=std.shape).as_in_context(ctx)
        return mu + eps * std

    def decode(self, z):
        h3 = nd.Activation(self.fc3(z), 'relu')
        return nd.Activation(self.fc4(h3), 'sigmoid')

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
model.collect_params().initialize(mx.init.Uniform(1 / np.sqrt(400)), ctx=ctx)

reconstruction_function = g.loss.L2Loss()


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    BCE = nd.sum(BCE)
    # loss = 0.5 * sum(1 - log(sigma^2) + mu^2 + sigma^2)
    KLD_element = (nd.power(mu, 2) + nd.exp(logvar)) * (-1) + 1 + logvar
    KLD = nd.sum(KLD_element) * (-0.5)
    # KLD_element = nd.exp(logvar) + nd.power(mu, 2) - logvar - 1
    # KLD = nd.sum(KLD_element) * 0.5
    # KL divergence
    return BCE + KLD


optimizer = g.Trainer(model.collect_params(), 'adam',
                      {'learning_rate': learning_rate})

for epoch in range(num_epochs):
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        img, _ = data
        batch = img.shape[0]
        img = img.reshape((batch, -1))
        img = img.as_in_context(ctx)
        with mx.autograd.record():
            recon_batch, mu, logvar = model(img)
            loss = loss_function(recon_batch, img, mu, logvar)
        loss.backward()
        train_loss += loss.asscalar()
        optimizer.step(batch)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch,
                len(dataset), 100. * batch_idx / int(
                    len(dataset) / batch_size), loss.asscalar() / batch))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(dataset)))
    if epoch % 10 == 0:
        save = to_img(torch.FloatTensor(recon_batch.asnumpy()))
        save_image(save, './vae_img/image_{}.png'.format(epoch))

model.save_params('./vae.params')
