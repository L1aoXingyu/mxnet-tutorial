import logging
import sys

import mxnet as mx

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-15s %(message)s',
                    stream=sys.stdout)

# define hyperparameters
batch_size = 32
learning_rate = 1e-3
epochs = 100
step = 300
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

# define dataset and dataloader
mnist = mx.test_utils.get_mnist()

train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True,
                               data_name='data', label_name='softmax_label')
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size * 2, shuffle=False,
                              data_name='data', label_name='softmax_label')

# define symbol
data = mx.sym.var('data')  # (bs, 1, 28, 28)
label = mx.sym.var('softmax_label')
data = mx.sym.Flatten(data)  # (bs, 28*28)
fc = mx.sym.FullyConnected(data, num_hidden=10, name='fc')
logist = mx.sym.SoftmaxOutput(fc, label=label, name='softmax')

metric_list = [mx.metric.Accuracy(output_names=['softmax_output'], label_names=['softmax_label']),
               mx.metric.CrossEntropy(output_names=['softmax_output'], label_names=['softmax_label'])]
eval_metrics = mx.metric.CompositeEvalMetric(metric_list)

mod = mx.mod.Module(logist, data_names=['data'], label_names=['softmax_label'], context=ctx)
mod.fit(
    train_data=train_iter,
    eval_data=test_iter,
    optimizer='sgd',
    optimizer_params={'learning_rate': 0.01},
    batch_end_callback=mx.callback.Speedometer(batch_size, 500),
    epoch_end_callback=mx.callback.do_checkpoint('logistic', 10),
    eval_metric=eval_metrics,
    num_epoch=epochs
)
