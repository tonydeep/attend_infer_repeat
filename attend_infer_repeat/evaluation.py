import os.path as osp
import time

import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def rect(bbox, c=None, facecolor='none', label=None, ax=None, line_width=1):
    r = Rectangle((bbox[1], bbox[0]), bbox[3], bbox[2], linewidth=line_width,
                  edgecolor=c, facecolor=facecolor, label=label)

    if ax is not None:
        ax.add_patch(r)
    return r


def rect_stn(ax, width, height, stn_params, c=None, line_width=3):
    sx, sy, tx, ty = stn_params
    x = width * (1. - sx + tx) / 2
    y = height * (1. - sy + ty) / 2
    bbox = [y - .5, x - .5, height * sy, width * sx]
    rect(bbox, c, ax=ax, line_width=line_width)


class ProgressFig(object):
    _BBOX_COLORS = 'rgbymcw'

    def __init__(self, air, sess, checkpoint_dir=None, n_samples=10, seq_n_samples=3, dpi=300,
                 fig_scale=1.5):

        self.air = air
        self.sess = sess
        self.checkpoint_dir = checkpoint_dir
        self.n_samples = n_samples
        self.seq_n_samples = seq_n_samples
        self.dpi = dpi
        self.fig_scale = fig_scale

        self.n_steps = self.air.max_steps
        self.n_timesteps = getattr(self.air, 'n_timesteps', None)
        self.with_time = self.n_timesteps is not None
        self.height, self.width = air.img_size[:2]

    def plot_all(self, global_step=None, save=True):
        self.plot_still(global_step, save)
        if self.with_time:
            self.plot_seq(global_step, save)

    def plot_still(self, global_step=None, save=True):

        xx, pred_canvas, pred_crop, prob, pres, w = self._air_outputs(single_timestep=True)
        fig, axes = self._make_fig(self.n_steps + 2, self.n_samples)

        # ground-truth
        for i, ax in enumerate(axes[0]):
            ax.imshow(xx[i], cmap='gray', vmin=0, vmax=1)

        # reconstructions with marked steps
        for i, ax in enumerate(axes[1]):
            ax.imshow(pred_canvas[i], cmap='gray', vmin=0, vmax=1)
            for j, c in zip(xrange(self.n_steps), self._BBOX_COLORS):
                if pres[i, j] > .5:
                    self._rect(ax, w[i, j], c)

        # glimpses
        for i, ax_row in enumerate(axes[2:]):
            for j, ax in enumerate(ax_row):
                ax.imshow(pres[j, i] * pred_crop[j, i], cmap='gray')
                ax.set_title('{:d} with p({:d}) = {:.02f}'.format(int(pres[j, i]), i + 1, prob[j, i]),
                             fontsize=4 * self.fig_scale)

                if pres[j, i] > .5:
                    for spine in 'bottom top left right'.split():
                        ax.spines[spine].set_color(self._BBOX_COLORS[i])
                        ax.spines[spine].set_linewidth(2.)

                ax_row[0].set_ylabel('glimpse #{}'.format(i + 1))

        for ax in axes.flatten():
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

        axes[0, 0].set_ylabel('ground-truth')
        axes[1, 0].set_ylabel('reconstruction')

        self._maybe_save_fig(fig, global_step, save, 'still_fig')

    def plot_seq(self, global_step=None, save=True):

        xx, pred_canvas, pred_crop, prob, pres, w = self._air_outputs(n_samples=self.seq_n_samples)
        fig, axes = self._make_fig(2 * self.seq_n_samples, self.n_timesteps)
        axes = axes.reshape((2 * self.seq_n_samples, self.n_timesteps))
        n = np.random.randint(xx.shape[1])

        for t, ax in enumerate(axes.T):
            for n in xrange(self.seq_n_samples):
                pres_time = pres[t, n, :]
                ax[2 * n].imshow(xx[t, n], cmap='gray', vmin=0., vmax=1.)
                ax[2 * n + 1].set_title(str(int(np.round(pres_time.sum()))), fontsize=6 * self.fig_scale)
                ax[2 * n + 1].imshow(pred_canvas[t, n], cmap='gray', vmin=0., vmax=1.)
                for i, (p, c) in enumerate(zip(pres_time, self._BBOX_COLORS)):
                    if p > .5:
                        self._rect(ax[2 * n + 1], w[t, n, i], c, line_width=1.)

        for n in xrange(self.seq_n_samples):
            axes[2 * n, 0].set_ylabel('gt #{:d}'.format(n))
            axes[2 * n + 1, 0].set_ylabel('rec #{:d}'.format(n))

        for a in axes.flatten():
            a.grid(False)
            a.set_xticks([])
            a.set_yticks([])

        self._maybe_save_fig(fig, global_step, save, 'seq_fig')

    def _maybe_save_fig(self, fig, global_step, save, root_name):
        if save and self.checkpoint_dir is not None:
            fig_name = osp.join(self.checkpoint_dir, '{}_{}.png'.format(root_name, global_step))
            fig.savefig(fig_name, dpi=self.dpi)
            plt.close(fig)

    def _air_outputs(self, n_samples=None, single_timestep=False):

        if n_samples is None:
            n_samples = self.n_samples

        if not getattr(self, '_air_tensors', None):
            names = 'canvas glimpse posterior_step_prob presence where'.split()
            tensors = [getattr(self.air, 'resampled_' + name, getattr(self.air, name)) for name in names]
            tensors[2] = tensors[2][..., 1:]
            self._air_tensors = [self.air.obs] + tensors

        res = self.sess.run(self._air_tensors)
        bs = np.random.choice(self.air.batch_size, size=n_samples, replace=False)
        ts = slice(None)
        if single_timestep:
            n_timesteps = res[0].shape[0]
            ts = np.random.choice(n_timesteps, size=self.n_samples, replace=True)

        for i, r in enumerate(res):
            if self.with_time:
                res[i] = r[ts, bs]
            else:
                res[i] = r[bs]

            if res[i].shape[-1] == 1:
                res[i] = res[i][..., 0]

        return res

    def _rect(self, ax, coords, color, line_width=2.):
        rect_stn(ax, self.width, self.height, coords, color, line_width=2.)

    def _make_fig(self, h, w, *args, **kwargs):
        figsize = self.fig_scale * np.asarray((w, h))
        return plt.subplots(h, w, figsize=figsize)


def make_logger(air, sess, summary_writer, train_tensor, n_train_samples, test_tensor, n_test_samples):
    exprs = {
        'nelbo': air.nelbo,
        'rec_loss': air.rec_loss,
        'num_step_acc': air.num_step_accuracy,
        'num_step': air.num_step,
        'kl_div': air.kl_div
    }

    additional_exprs = {
        'nums_xe': 'nums_xe',
        'kl_num_steps': 'kl_num_steps',
        'kl_what': 'kl_what',
        'kl_where': 'kl_where',
        'kl_scale': 'kl_scale',
        'kl_shift': 'kl_shift',
        'baseline_loss': 'baseline_loss',
        'reinforce_loss': 'reinforce_loss',
        'l2_loss': 'l2_loss',
        'proxy_loss': 'proxy_loss',
        # 'imp_weight': 'num_steps_learning_signal'
    }

    skipped = False
    for k, v in additional_exprs.iteritems():
        try:
            exprs[k] = getattr(air, v)
        except AttributeError:
            if not skipped:
                skipped = True
                print 'make_logger: unable to log all expressions:'
            print '\tSkipping {}'.format(k)

    train_log = make_expr_logger(sess, summary_writer, n_train_samples // air.batch_size, exprs, name='train')

    data_dict = {
        train_tensor['imgs']: test_tensor['imgs'],
        train_tensor['nums']: test_tensor['nums']
    }
    test_log = make_expr_logger(sess, summary_writer, n_test_samples // air.batch_size, exprs, name='test',
                                data_dict=data_dict)

    def log(train_itr, **kwargs):
        train_log(train_itr, **kwargs)
        test_log(train_itr, **kwargs)
        print

    return log


def make_expr_logger(sess, writer, num_batches, expr_dict, name, data_dict=None,
                     constants_dict=None, measure_time=True):
    """

    :param sess:
    :param writer:
    :param num_batches:
    :param expr:
    :param name:
    :param data_dict:
    :param constants_dict:
    :return:
    """

    tags = {k: '/'.join((k, name)) for k in expr_dict}
    data_name = 'Data {}'.format(name)
    log_string = ', '.join((''.join((k + ' = {', k, ':.4f}')) for k in expr_dict))
    log_string = ' '.join(('Step {},', data_name, log_string))

    if measure_time:
        log_string += ', eval time = {:.4}s'

        def make_log_string(itr, l, t):
            return log_string.format(itr, t, **l)
    else:
        def make_log_string(itr, l, t):
            return log_string.format(itr, **l)

    def log(itr, l, t):
        try:
            return make_log_string(itr, l, t)
        except ValueError as err:
            print err.message
            print '\tLogging items'
            for k, v in l.iteritems():
                print '{}: {}'.format(k, type(v))

    def logger(itr=0, num_batches_to_eval=None, write=True):
        l = {k: 0. for k in expr_dict}
        start = time.time()
        if num_batches_to_eval is None:
            num_batches_to_eval = num_batches

        for i in xrange(num_batches_to_eval):
            if data_dict is not None:
                vals = sess.run(data_dict.values())
                feed_dict = {k: v for k, v in zip(data_dict.keys(), vals)}
                if constants_dict:
                    feed_dict.update(constants_dict)
            else:
                feed_dict = constants_dict

            r = sess.run(expr_dict, feed_dict)
            for k, v in r.iteritems():
                l[k] += v

        for k, v in l.iteritems():
            l[k] /= num_batches_to_eval
        t = time.time() - start
        print log(itr, l, t)

        if write:
            log_values(writer, itr, [tags[k] for k in l.keys()], l.values())

        return l

    return logger


def log_ratio(var_tuple, name='ratio', eps=1e-8):
    """

    :param var_tuple:
    :param name:
    :param which_name:
    :param eps:
    :return:
    """
    a, b = var_tuple
    ratio = tf.reduce_mean(abs(a) / (abs(b) + eps))
    tf.summary.scalar(name, ratio)


def log_norm(expr_list, name):
    """

    :param expr_list:
    :param name:
    :return:
    """
    n_elems = 0
    norm = 0.
    for e in nest.flatten(expr_list):
        n_elems += tf.reduce_prod(tf.shape(e))
        norm += tf.reduce_sum(e ** 2)
    norm /= tf.to_float(n_elems)
    tf.summary.scalar(name, norm)
    return norm


def log_values(writer, itr, tags=None, values=None, dict=None):
    if dict is not None:
        assert tags is None and values is None
        tags = dict.keys()
        values = dict.values()
    else:

        if not nest.is_sequence(tags):
            tags, values = [tags], [values]

        elif len(tags) != len(values):
            raise ValueError('tag and value have different lenghts:'
                             ' {} vs {}'.format(len(tags), len(values)))

    for t, v in zip(tags, values):
        summary = tf.Summary.Value(tag=t, simple_value=v)
        summary = tf.Summary(value=[summary])
        writer.add_summary(summary, itr)


def gradient_summaries(gvs, norm=True, ratio=True, histogram=True):
    """Register gradient summaries.

    Logs the global norm of the gradient, ratios of gradient_norm/uariable_norm and
    histograms of gradients.

    :param gvs: list of (gradient, variable) tuples
    :param norm: boolean, logs norm of the gradient if True
    :param ratio: boolean, logs ratios if True
    :param histogram: boolean, logs gradient histograms if True
    """

    with tf.name_scope('grad_summary'):
        if norm:
            grad_norm = tf.global_norm([gv[0] for gv in gvs])
            tf.summary.scalar('grad_norm', grad_norm)

        for g, v in gvs:
            var_name = v.name.split(':')[0]
            if g is None:
                print 'Gradient for variable {} is None'.format(var_name)
                continue

            if ratio:
                log_ratio((g, v), '/'.join(('grad_ratio', var_name)))

            if histogram:
                tf.summary.histogram('/'.join(('grad_hist', var_name)), g)
