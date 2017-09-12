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
num_epoch = 100
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
discriminator = g.nn.HybridSequential(prefix='dis_')
with discriminator.name_scope():
    discriminator.add(g.nn.Dense(256))
    discriminator.add(g.nn.LeakyReLU(0.2))
    discriminator.add(g.nn.Dense(256))
    discriminator.add(g.nn.LeakyReLU(0.2))
    discriminator.add(g.nn.Dense(1, activation='sigmoid'))

discriminator.hybridize()
# Generator
generator = g.nn.HybridSequential(prefix='gen_')
with generator.name_scope():
    generator.add(g.nn.Dense(256, activation='relu'))
    generator.add(g.nn.Dense(256, activation='relu'))
    generator.add(g.nn.Dense(784, activation='tanh'))

generator.hybridize()

discriminator.collect_params().initialize(mx.init.Xavier(), ctx)
generator.collect_params().initialize(mx.init.Xavier(), ctx)
# Binary cross entropy loss and optimizer
bce = g.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

d_optimizer = g.Trainer(discriminator.collect_params(), 'adam',
                        {'learning_rate': 0.0003})
g_optimizer = g.Trainer(generator.collect_params(), 'adam',
                        {'learning_rate': 0.0003})

# Start training
for epoch in range(num_epoch):
    for i, (img, _) in enumerate(dataloader):
        num_img = img.shape[0]
        # =================train discriminator
        img = img.reshape((num_img, -1))
        real_img = img.as_in_context(ctx)
        real_label = nd.ones(shape=[num_img], ctx=ctx)
        fake_label = nd.zeros(shape=[num_img], ctx=ctx)

        # compute loss of real_img
        with mx.autograd.record():
            real_out = discriminator(real_img)
            d_loss_real = bce(real_out, real_label)
        real_scores = real_out  # closer to 1 means better

        # compute loss of fake_img
        z = nd.random_normal(
            loc=0, scale=1, shape=[num_img, z_dimension], ctx=ctx)
        with g.autograd.record():
            fake_img = generator(z)
            fake_out = discriminator(fake_img)
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
            fake_img = generator(z)
            output = discriminator(fake_img)
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

discriminator.save_params('./dis.params')
generator.save_params('./gen.params')
