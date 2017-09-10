import os

import mxnet as mx
import torch
from mxnet import gluon as g
from mxnet import ndarray as nd
from torchvision.utils import save_image

if not os.path.exists('./img'):
    os.mkdir('./img')


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


batch_size = 128
num_epoch = 5
z_dimension = 100
ctx = mx.gpu()


# Image processing
def img_transform(data, label):
    data = (data.astype('float32') / 255 - 0.5) / 0.5
    return data.transpose((2, 0, 1)), label.astype('float32')


# MNIST dataset
mnist = g.data.vision.MNIST(transform=img_transform)
# Data loader
dataloader = g.data.DataLoader(
    dataset=mnist, batch_size=batch_size, shuffle=True)


# Discriminator
class discriminator(g.HybridBlock):
    def __init__(self):
        super(discriminator, self).__init__()
        with self.name_scope():
            self.conv1 = g.nn.HybridSequential(prefix='conv1_')
            with self.conv1.name_scope():
                self.conv1.add(g.nn.Conv2D(32, 5, padding=2))
                self.conv1.add(g.nn.LeakyReLU(0.2))
                self.conv1.add(g.nn.AvgPool2D(2, 2))

            self.conv2 = g.nn.HybridSequential(prefix='conv2_')
            with self.conv2.name_scope():
                self.conv2.add(g.nn.Conv2D(64, 5, padding=2))
                self.conv2.add(g.nn.LeakyReLU(0.2))
                self.conv2.add(g.nn.AvgPool2D(2, 2))

            self.fc = g.nn.HybridSequential(prefix='fc_')
            with self.fc.name_scope():
                self.fc.add(g.nn.Flatten())
                self.fc.add(g.nn.Dense(1024))
                self.fc.add(g.nn.LeakyReLU(0.2))
                self.fc.add(g.nn.Dense(1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x


d = discriminator()
d.hybridize()


# Generator
class generator(g.HybridBlock):
    def __init__(self, num_feature):
        super(generator, self).__init__()
        with self.name_scope():
            self.fc = g.nn.Dense(num_feature)

            self.br = g.nn.HybridSequential(prefix='batch_relu_')
            with self.br.name_scope():
                self.br.add(g.nn.BatchNorm())
                self.br.add(g.nn.Activation('relu'))

            self.downsample = g.nn.HybridSequential(prefix='ds_')
            with self.downsample.name_scope():
                self.downsample.add(g.nn.Conv2D(50, 3, strides=1, padding=1))
                self.downsample.add(g.nn.BatchNorm())
                self.downsample.add(g.nn.Activation('relu'))
                self.downsample.add(g.nn.Conv2D(25, 3, strides=1, padding=1))
                self.downsample.add(g.nn.BatchNorm())
                self.downsample.add(g.nn.Activation('relu'))
                self.downsample.add(
                    g.nn.Conv2D(1, 2, strides=2, activation='tanh'))

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape((x.shape[0], 1, 56, 56))
        x = self.br(x)
        x = self.downsample(x)
        return x


ge = generator(3136)
ge.hybridize()

d.collect_params().initialize(mx.init.Xavier(), ctx)
ge.collect_params().initialize(mx.init.Xavier(), ctx)
# Binary cross entropy loss and optimizer
bce = g.loss.SigmoidBinaryCrossEntropyLoss()

d_optimizer = g.Trainer(d.collect_params(), 'adam', {'learning_rate': 0.0003})
g_optimizer = g.Trainer(ge.collect_params(), 'adam', {'learning_rate': 0.0003})

# Start training
for epoch in range(num_epoch):
    for i, (img, _) in enumerate(dataloader):
        num_img = img.shape[0]
        # =================train discriminator
        real_img = img.as_in_context(ctx)
        real_label = nd.ones(shape=[num_img], ctx=ctx)
        fake_label = nd.zeros(shape=[num_img], ctx=ctx)

        # compute loss of real_img
        with g.autograd.record():
            real_out = d(real_img)
            d_loss_real = bce(real_out, real_label)
        real_scores = real_out  # closer to 1 means better

        # compute loss of fake_img
        z = nd.random_normal(
            loc=0, scale=1, shape=[num_img, z_dimension], ctx=ctx)
        with g.autograd.record():
            fake_img = ge(z)
            fake_out = d(fake_img)
            d_loss_fake = bce(fake_out, fake_label)
        fake_scores = fake_out  # closer to 0 means better

        # bp and optimize
        with g.autograd.record():
            d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step(num_img)

        # ===============train generator
        # compute loss of fake_img
        with g.autograd.record():
            fake_img = ge(z)
            output = d(fake_img)
            g_loss = bce(output, real_label)

        # bp and optimize
        g_loss.backward()
        g_optimizer.step(num_img)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                  'D real: {:.6f}, D fake: {:.6f}'.format(
                      epoch, num_epoch,
                      nd.mean(d_loss).asscalar(),
                      nd.mean(g_loss).asscalar(),
                      nd.mean(real_scores).asscalar(),
                      nd.mean(fake_scores).asscalar()))
    if epoch == 0:
        real_images = to_img(torch.FloatTensor(real_img.asnumpy()))
        save_image(real_images, './img/real_images.png')

    fake_images = to_img(torch.FloatTensor(fake_img.asnumpy()))
    save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))

d.save_params('./dis.params')
ge.save_params('./gen.params')
