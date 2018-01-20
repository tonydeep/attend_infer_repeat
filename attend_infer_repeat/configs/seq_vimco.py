import tensorflow as tf
import sonnet as snt

from attend_infer_repeat.mnist_model import APDRonMNIST
from attend_infer_repeat.experiment_tools import optimizer_from_string
from attend_infer_repeat.ops import maybe_getattr
from attend_infer_repeat.model import APDRModel
from attend_infer_repeat import tf_flags as flags

flags.DEFINE_float('step_bias', 1., '')
flags.DEFINE_float('transform_var_bias', -3., '')
flags.DEFINE_float('learning_rate', 1e-5, '')
flags.DEFINE_float('output_multiplier', .25, '')
flags.DEFINE_float('init_step_success_prob', 1. - 1e-7, '')
flags.DEFINE_float('final_step_success_prob', 1e-5, '')
flags.DEFINE_float('n_anneal_steps_loss', 1e3, '')
flags.DEFINE_float('min_glimpse_size', 0., '')
flags.DEFINE_float('prop_logit_bias', 3., '')
flags.DEFINE_float('constant_prop_prior', 0., '')

flags.DEFINE_integer('n_iw_samples', 5, '')
flags.DEFINE_integer('n_steps_per_image', 3, '')
flags.DEFINE_boolean('importance_resample', False, '')
flags.DEFINE_boolean('decode_prop', False, '')
flags.DEFINE_boolean('propagate_disc_what', False, '')
flags.DEFINE_boolean('internal_decode', False, '')
flags.DEFINE_boolean('discover_only_t0', False, '')


flags.DEFINE_string('opt', '', '')
flags.DEFINE_string('transition', 'VanillaRNN', '')
flags.DEFINE_string('time_transition', 'GRU', '')
flags.DEFINE_string('prior_transition', 'GRU', '')

flags.DEFINE_boolean('anneal_prior', True, '')
flags.DEFINE_float('anneal_temp', 1., '')
flags.DEFINE_integer('anneal_iter', 0, '')

flags.DEFINE_boolean('relation_embed', False, '')

flags.DEFINE_float('output_std', .3, '')
flags.DEFINE_boolean('learnable_output_std', False, '')


def load(img, num):

    f = flags.FLAGS

    n_hidden = 32 * 8
    n_layers = 2
    n_hiddens = [n_hidden] * n_layers

    transition = maybe_getattr(snt, f.transition)
    time_transition = maybe_getattr(snt, f.time_transition)
    prior_transition = maybe_getattr(snt, f.prior_transition)

    class APDRwithVIMCO(APDRonMNIST):
        output_std = f.output_std
        learnable_output_std = f.learnable_output_std

        importance_resample = f.importance_resample
        init_step_success_prob = f.init_step_success_prob
        final_step_success_prob = f.final_step_success_prob
        n_anneal_steps_loss = f.n_anneal_steps_loss
        prop_logit_bias = f.prop_logit_bias
        transition_class = transition
        time_transition_class = time_transition
        prior_rnn_class = prior_transition
        decode_prop = f.decode_prop
        constant_prop_prior = f.constant_prop_prior
        propagate_disc_what = f.propagate_disc_what
        anneal_prior = f.anneal_prior
        anneal_iter = f.anneal_iter
        anneal_temp = f.anneal_temp
        relation_embed = f.relation_embed
        internal_decode = f.internal_decode
        discover_only_t0 = f.discover_only_t0

    air = APDRwithVIMCO(img,
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
    )

    kwargs = dict(learning_rate=f.learning_rate, nums=num)
    if f.opt:
        opt, opt_kwargs = optimizer_from_string(f.opt, build=False)
        kwargs['optimizer'] = opt
        kwargs['opt_kwargs'] = opt_kwargs

    train_step, global_step = air.train_step(**kwargs)

    return air, train_step, global_step
