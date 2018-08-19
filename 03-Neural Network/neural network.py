import logging
import sys

import mxnet as mx

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)-15s %(message)s",
                    stream=sys.stdout)

# define hyperparameters
batch_size = 32
learning_rate = 1e-2
epochs = 100
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

mnist = mx.test_utils.get_mnist()
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True,
                               label_name='softmax_label')
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], 2 * batch_size, shuffle=False,
                             label_name='softmax_label')

# define symbol
data = mx.sym.var('data')
label = mx.sym.var('softmax_label')

data = mx.sym.Flatten(data)  # (bs, 28*28)
fc1 = mx.sym.FullyConnected(data, num_hidden=300, name='fc1')
act1 = mx.sym.Activation(fc1, act_type='relu', name='relu1')
fc2 = mx.sym.FullyConnected(act1, num_hidden=100, name='fc2')
act2 = mx.sym.Activation(fc2, act_type='relu', name='relu2')
fc3 = mx.sym.FullyConnected(act2, num_hidden=10, name='fc3')
net = mx.sym.SoftmaxOutput(fc3, label=label, name='softmax')

mod = mx.mod.Module(net, data_names=['data'], label_names=['softmax_label'], context=ctx)

metric_list = [mx.metric.CrossEntropy(output_names=['softmax_output'], label_names=['softmax_label']),
               mx.metric.Accuracy(output_names=['softmax_output'], label_names=['softmax_label'])]
eval_metrics = mx.metric.CompositeEvalMetric(metric_list)

mod.fit(
    train_data=train_iter,
    eval_data=val_iter,
    initializer=mx.init.MSRAPrelu(),
    optimizer='sgd',
    optimizer_params={'learning_rate': 0.1},
    eval_metric=eval_metrics,
    batch_end_callback=mx.callback.Speedometer(batch_size, 500),
    epoch_end_callback=mx.callback.do_checkpoint('nn', 10),
    num_epoch=epochs
)
