
# coding: utf-8

# In[ ]:

from os import path as osp
import tensorflow as tf

from evaluation import ProgressFig, make_logger
from experiment_tools import load, init_checkpoint, parse_flags, print_flags, set_flags_if_notebook, is_notebook
from attend_infer_repeat import tf_flags, evaluation

import matplotlib.pyplot as plt


# In[ ]:

import sys
sys.path.append('../')


# In[ ]:

# Define flags

tf_flags.DEFINE_string('data_config', 'configs/seq_mnist_data.py', '')
tf_flags.DEFINE_string('model_config', 'configs/seq_vimco.py', '')
tf_flags.DEFINE_string('results_dir', '../checkpoints', '')
tf_flags.DEFINE_string('run_name', 'test_run', '')

tf_flags.DEFINE_integer('batch_size', 64, '')

tf_flags.DEFINE_integer('summary_every', 1000, '')
tf_flags.DEFINE_integer('log_every', 5000, '')
tf_flags.DEFINE_integer('save_every', 5000, '')
tf_flags.DEFINE_integer('max_train_iter', int(3 * 1e5), '')
tf_flags.DEFINE_boolean('resume', False, '')
tf_flags.DEFINE_boolean('log_at_start', False, '')

tf_flags.DEFINE_float('eval_size_fraction', .01, '')


# In[ ]:

set_flags_if_notebook(
#     data_config='configs/static_mnist_data.py',
    seq_len=5,
    n_steps_per_image=3,
    n_iw_samples=5,
#     
    log_every=100,
    eval_size_fraction=0.01,
#     
    learning_rate=1e-5,
    importance_resample=True,
    step_bias=1.,
#     init_step_success_prob=.9,
#     
#     resume=True
)

# Parse flags
parse_flags()
F = tf_flags.FLAGS


# In[ ]:

# Prepare enviornment
logdir = osp.join(F.results_dir, F.run_name)
logdir, flags, resume_checkpoint = init_checkpoint(logdir, F.data_config, F.model_config, F.resume)
checkpoint_name = osp.join(logdir, 'model.ckpt')


# In[ ]:

# Build the graph
tf.reset_default_graph()
data_dict = load(F.data_config, F.batch_size)
air, train_step, global_step = load(F.model_config, img=data_dict.train_img, num=data_dict.train_num)

print_flags()


# In[ ]:

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# In[ ]:

sess.run(tf.global_variables_initializer())


# In[ ]:

saver = tf.train.Saver()
if resume_checkpoint is not None:
    print "Restoring checkpoint from '{}'".format(resume_checkpoint)
    saver.restore(sess, resume_checkpoint)


# In[ ]:

summary_writer = tf.summary.FileWriter(logdir, sess.graph)
all_summaries = tf.summary.merge_all()


# In[ ]:

# Logging
ax = data_dict['axes']['imgs']
factor = F.eval_size_fraction
train_batches, valid_batches = [int(data_dict[k]['imgs'].shape[ax] * factor) for k in ('train_data', 'valid_data')]
log = make_logger(air, sess, summary_writer, data_dict.train_tensors,
                  train_batches, data_dict.valid_tensors, valid_batches)

progress_fig = ProgressFig(air, sess, logdir, seq_n_samples=16)


# In[ ]:

# progress_fig.plot_all(save=False)


# In[ ]:

train_itr = sess.run(global_step)
print 'Starting training at iter = {}'.format(train_itr)


if F.log_at_start or train_itr == 0:
    log(train_itr)
    if not is_notebook():
        progress_fig.plot_all(train_itr)

while train_itr < F.max_train_iter:

    train_itr, _ = sess.run([global_step, train_step])

    if train_itr % F.summary_every == 0:
        summaries = sess.run(all_summaries)
        summary_writer.add_summary(summaries, train_itr)

    if train_itr % F.log_every == 0:
        log(train_itr)

    if train_itr % F.save_every == 0:
        saver.save(sess, checkpoint_name, global_step=train_itr)
        progress_fig.plot_all(train_itr)


# In[ ]:

# progress_fig.plot_all(save=False)

