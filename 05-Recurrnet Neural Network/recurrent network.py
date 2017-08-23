import mxnet as mx
import mxnet.gluon as g

ctx = mx.gpu()
net = g.nn.Sequential()
with net.name_scope():
    net.add(g.nn.Dense(10, in_units=100))

net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
