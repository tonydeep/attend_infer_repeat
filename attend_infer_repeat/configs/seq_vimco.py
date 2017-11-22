import tensorflow as tf
import sonnet as snt

from attend_infer_repeat.mnist_model import SeqAIRonMNIST, KLBySamplingMixin
from attend_infer_repeat.experiment_tools import optimizer_from_string
from attend_infer_repeat.ops import maybe_getattr

flags = tf.flags

tf.flags.DEFINE_float('step_bias', 1., '')
tf.flags.DEFINE_float('transform_var_bias', -3., '')
tf.flags.DEFINE_float('learning_rate', 1e-5, '')
tf.flags.DEFINE_float('output_multiplier', .25, '')
tf.flags.DEFINE_float('init_step_success_prob', 1. - 1e-7, '')
tf.flags.DEFINE_float('final_step_success_prob', 1e-5, '')
tf.flags.DEFINE_float('n_anneal_steps_loss', 1e3, '')
tf.flags.DEFINE_float('min_glimpse_size', 0., '')
tf.flags.DEFINE_float('where_prior_scale', 1., '')
tf.flags.DEFINE_integer('n_iw_samples', 5, '')
tf.flags.DEFINE_integer('n_steps_per_image', 3, '')
tf.flags.DEFINE_boolean('importance_resample', False, '')
tf.flags.DEFINE_boolean('condition_on_prev', False, '')
tf.flags.DEFINE_boolean('condition_on_latents', False, '')
tf.flags.DEFINE_boolean('condition_on_rnn_output', False, '')
tf.flags.DEFINE_boolean('transition_only_on_object', False, '')
tf.flags.DEFINE_boolean('prior_around_prev', False, '')
tf.flags.DEFINE_string('opt', '', '')
tf.flags.DEFINE_string('transition', 'LSTM', '')
tf.flags.DEFINE_string('time_transition', None, '')


def load(img, num):

    f = tf.flags.FLAGS

    n_hidden = 32 * 8
    n_layers = 2
    n_hiddens = [n_hidden] * n_layers

    transition = maybe_getattr(snt, f.transition)
    time_transition = maybe_getattr(snt, f.time_transition)

    class SeqAIRwithVIMCO(SeqAIRonMNIST, KLBySamplingMixin):
        importance_resample = f.importance_resample
        init_step_success_prob = f.init_step_success_prob
        final_step_success_prob = f.final_step_success_prob
        n_anneal_steps_loss = f.n_anneal_steps_loss
        where_prior_scale = f.where_prior_scale
        transition_class = transition
        time_transition_class = time_transition

    air = SeqAIRwithVIMCO(img,
                      max_steps=f.n_steps_per_image,
                      inpt_encoder_hidden=n_hiddens,
                      glimpse_encoder_hidden=n_hiddens,
                      glimpse_decoder_hidden=n_hiddens,
                      transform_estimator_hidden=n_hiddens,
                      steps_pred_hidden=[128, 64],
                      transform_var_bias=f.transform_var_bias,
                      min_glimpse_size=f.min_glimpse_size,
                      step_bias=f.step_bias,
                      iw_samples=f.n_iw_samples,
                      output_multiplier=f.output_multiplier,
                      condition_on_prev=f.condition_on_prev,
                      condition_on_rnn_output=f.condition_on_rnn_output,
                      condition_on_latents=f.condition_on_latents,
                      prior_around_prev=f.prior_around_prev,
                      transition_only_on_object=f.transition_only_on_object)

    kwargs = dict(learning_rate=f.learning_rate, nums=num)
    if f.opt:
        opt, opt_kwargs = optimizer_from_string(f.opt, build=False)
        kwargs['optimizer'] = opt
        kwargs['opt_kwargs'] = opt_kwargs

    train_step, global_step = air.train_step(**kwargs)

    return air, train_step, global_step
