from functools import partial

import sonnet as snt
import tensorflow as tf

from modules import Encoder, Decoder, StochasticTransformParam, StepsPredictor
from ops import anneal_weight
from model import APDRModel


class MNISTPriorMixin(object):
    init_step_success_prob = 1. - 1e-7
    final_step_success_prob = 1e-5

    def _geom_success_prob(self, **kwargs):

        hold_init = 1e3
        steps_div = 1e4
        anneal_steps = 1e5
        global_step = tf.train.get_or_create_global_step()
        steps_prior_success_prob = anneal_weight(self.init_step_success_prob, self.final_step_success_prob, 'exp', global_step,
                                                     anneal_steps, hold_init, steps_div)
        self.steps_prior_success_prob = steps_prior_success_prob
        return self.steps_prior_success_prob


class APDRonMNIST(APDRModel, MNISTPriorMixin):
    """Implements AIR for the MNIST dataset"""
    transition_class = snt.LSTM

    def __init__(self, obs, glimpse_size=(20, 20),
                 inpt_encoder_hidden=[256]*2,
                 glimpse_encoder_hidden=[256]*2,
                 glimpse_decoder_hidden=[252]*2,
                 transform_estimator_hidden=[256]*2,
                 steps_pred_hidden=[50]*1,
                 transform_var_bias=-2.,
                 min_glimpse_size=0.,
                 step_bias=0.,
                 *args, **kwargs):

        self.transform_var_bias = tf.Variable(transform_var_bias, trainable=False, dtype=tf.float32,
                                                       name='transform_var_bias')
        self.min_glimpse_size = min_glimpse_size
        self.step_bias = tf.Variable(step_bias, trainable=False, dtype=tf.float32, name='step_bias')
        super(APDRonMNIST, self).__init__(
            *args,
            obs=obs,
            glimpse_size=glimpse_size,
            n_what=50,
            transition=self.transition_class(256),
            input_encoder=partial(Encoder, inpt_encoder_hidden),
            glimpse_encoder=partial(Encoder, glimpse_encoder_hidden),
            glimpse_decoder=partial(Decoder, glimpse_decoder_hidden),
            transform_estimator=partial(StochasticTransformParam, transform_estimator_hidden,
                                      scale_bias=self.transform_var_bias, min_glimpse_size=self.min_glimpse_size),
            steps_predictor=partial(StepsPredictor, steps_pred_hidden, self.step_bias),
            output_std=.3,
            **kwargs
        )
