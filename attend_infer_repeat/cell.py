import numpy as  np
import sonnet as snt
import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli, Normal, NormalWithSoftplusScale
from tensorflow.python.util import nest

from modules import SpatialTransformer, ParametrisedGaussian
from neural import MLP
import ops


class BaseAPDRCell(snt.RNNCore):
    _n_transform_param = 4
    _init_presence_value = 1.  # at the beginning we assume all objects were present
    _output_names = 'what what_loc what_scale where where_loc where_scale presence_prob presence presence_logit'.split()

    _latent_name_to_idx = dict(
        what=0,
        where=3,
        pres_prob=6,
        pres=7,
        pres_logit=8,
    )

    def __init__(self, img_size, crop_size, n_what,
                 transition, input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                 decoder=None,
                 debug=False):
        """Creates the cell

        :param img_size: int tuple, size of the image
        :param crop_size: int tuple, size of the attention glimpse
        :param n_what: number of latent units describing the "what"
        :param transition: an RNN cell for maintaining the internal hidden state
        :param input_encoder: callable, encodes the original input image before passing it into the transition
        :param glimpse_encoder: callable, encodes the glimpse into latent representation
        :param transform_estimator: callabe, transforms the hidden state into parameters for the spatial transformer
        :param steps_predictor: callable, predicts whether to take a step
        :param debug: boolean, adds checks for NaNs in the inputs to distributions
        """

        super(BaseAPDRCell, self).__init__()
        self._img_size = img_size
        self._n_pix = np.prod(self._img_size)
        self._crop_size = crop_size
        self._n_what = n_what
        self._transition = transition
        self._n_hidden = int(self._transition.output_size[0])

        self._decoder = decoder

        self._debug = debug

        with self._enter_variable_scope():

            self._spatial_transformer = SpatialTransformer(img_size, crop_size)

            self._transform_estimator = transform_estimator()
            self._input_encoder = input_encoder()
            self._glimpse_encoder = glimpse_encoder()

            self._what_distrib = ParametrisedGaussian(n_what, scale_offset=0.5,
                                                      validate_args=self._debug, allow_nan_stats=not self._debug)
            self._steps_predictor = steps_predictor()

    @property
    def state_size(self):
        return [
            np.prod(self._img_size),  # image
            self._n_what,  # what
            self._n_transform_param,  # where
            1,  # presence
            self._transition.state_size,  # hidden state of the rnn
        ]

    @property
    def output_size(self):
        return [
            self._n_what,  # what code
            self._n_what,  # what loc
            self._n_what,  # what scale
            self._n_transform_param,  # where code
            self._n_transform_param,  # where loc
            self._n_transform_param,  # where scale
            1,  # presence prob
            1,  # presence
            1  # presence_logit
        ]

    @property
    def output_names(self):
        return self._output_names

    @staticmethod
    def outputs_by_name(hidden_outputs):
        return {n: o for n, o in zip(DiscoveryCell._output_names, hidden_outputs)}

    @staticmethod
    def extract_latents(hidden_outputs, key=None, skip=None):
        if key is not None and skip is not None:
            raise ValueError("Either `key' or `skip' have to be None, but both are not!")

        if skip is not None:
            key = (k for k in BaseAPDRCell._latent_name_to_idx.keys() if k not in nest.flatten(skip))
            latent_idx = sorted((BaseAPDRCell._latent_name_to_idx[k] for k in key))

        elif key is None:
            latent_idx = sorted(BaseAPDRCell._latent_name_to_idx.values())
        else:
            latent_idx = (BaseAPDRCell._latent_name_to_idx[k] for k in nest.flatten(key))

        if isinstance(hidden_outputs[0], tf.Tensor):
            latents = [hidden_outputs[i] for i in latent_idx]
        else:
            latents = [ops.extract_state(hidden_outputs, i) for i in latent_idx]

        if len(latents) == 1:
            latents = latents[0]
        return latents

    def initial_state(self, img, hidden_state=None):
        batch_size = img.get_shape().as_list()[0]

        if hidden_state is None:
            hidden_state = self._transition.initial_state(batch_size, tf.float32, trainable=True)

        where_code = tf.zeros([1, self._n_transform_param], dtype=tf.float32, name='where_init')
        what_code = tf.zeros([1, self._n_what], dtype=tf.float32, name='what_init')

        where_code, what_code = (tf.tile(i, (batch_size, 1)) for i in (where_code, what_code))

        flat_img = tf.reshape(img, (batch_size, self._n_pix))
        init_presence = tf.ones((batch_size, 1), dtype=tf.float32) * self._init_presence_value
        return [flat_img, what_code, where_code, init_presence, hidden_state]

    def _extract_and_encode_glimpse(self, img, where_code):
        cropped = self._spatial_transformer(img, logits=where_code)
        flat_crop = snt.BatchFlatten()(cropped)
        return self._glimpse_encoder(flat_crop)

    def _prepare_rnn_inputs(self, inpt, img, what, where, presence):
        transition_inpt = self._input_encoder(img)
        transition_inpt = [transition_inpt]
        if inpt is not None:
            transition_inpt += nest.flatten(inpt)

        if self._decoder is not None:
            transition_inpt += [what, where, presence]

        if len(transition_inpt) > 1:
            transition_inpt = tf.concat(transition_inpt, -1)
        else:
            transition_inpt = transition_inpt[0]

        return transition_inpt

    def _maybe_transition(self, is_allowed, output, old_state, new_state):
        """Implements sequence lengths

            input is an indicator vector, where one means that a transition is allowed while 0
            means there should be no transition. All outputs for disallowed transitions are set
            to zero, while the hidden state for the last allowed transition is propagated"""

        # set what, where, pres_prob, pres, pres_logit to zero
        for i in (0, 3, 6, 7, 8):
            output[i] *= is_allowed

        state = []
        for os, ns in zip(old_state, new_state):
            state.append(is_allowed * ns + (1 - is_allowed) * os)

        return output, state

    def _build(self, inpt, state):
        """Input is unused; it's only to force a maximum number of steps"""
        img_flat, what_code, where_code, presence, hidden_state = state
        inpt, is_allowed = self._parse_inpt(inpt, presence)
        img = tf.reshape(img_flat, (-1,) + tuple(self._img_size))

        with tf.variable_scope('rnn_inpt'):
            rnn_inpt = self._prepare_rnn_inputs(inpt, img, what_code, where_code, presence)
            if self._decoder is None:
                hidden_output, hidden_state = self._transition(rnn_inpt, hidden_state)
            else:
                hidden_output = MLP([self._n_hidden]*2, name='transition_MLP')(rnn_inpt)

        with tf.variable_scope('where'):
            where_code, where_loc, where_scale = self._compute_where(inpt, hidden_output)

        with tf.variable_scope('presence'):
            presence, presence_prob, presence_logit\
                = self._compute_presence(inpt, presence, hidden_output)

        with tf.variable_scope('what'):
            what_code, what_loc, what_scale = self._compute_what(inpt, img, where_code)

        if self._decoder is not None:
            params = [tf.expand_dims(i, 1) for i in (what_code, where_code, presence)]
            reconstruction, _ = self._decoder(*params)
            img_flat -= tf.reshape(reconstruction, tf.shape(img_flat))

        output = [what_code, what_loc, what_scale, where_code, where_loc, where_scale,
                  presence_prob, presence, presence_logit]
        new_state = [img_flat, what_code, where_code, presence, hidden_state]

        return self._maybe_transition(is_allowed, output, state, new_state)


class DiscoveryCell(BaseAPDRCell):

    def _parse_inpt(self, inpt, presence):
        inpt, is_allowed = inpt
        return inpt, is_allowed * presence

    def _compute_what(self, inpt, img, where_code):
        what_params = self._extract_and_encode_glimpse(img, where_code)
        what_distrib = self._what_distrib(what_params)
        return what_distrib.sample(), what_distrib.loc, what_distrib.scale

    def _compute_where(self, inpt, hidden_output):
        where_param = self._transform_estimator(hidden_output)
        loc, scale = where_param
        loc *= .1
        where_distrib = NormalWithSoftplusScale(scale, loc,
                                                validate_args=self._debug, allow_nan_stats=not self._debug)
        return where_distrib.sample(), where_distrib.loc, where_distrib.scale

    def _compute_presence(self, inpt, presence, hidden_output):
        presence_logit = self._steps_predictor(hidden_output)
        presence_prob = tf.nn.sigmoid(presence_logit)

        presence_distrib = Bernoulli(probs=presence_prob, dtype=tf.float32,
                                     validate_args=self._debug, allow_nan_stats=not self._debug)
        new_presence = presence_distrib.sample()
        presence *= new_presence

        return presence, presence_prob, presence_logit


class PropagationCell(BaseAPDRCell):
    _init_presence_value = 0.  # at the beginning we assume no objects

    def __init__(self, img_size, crop_size, n_what,
                 transition, input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                 latent_scale=1.0, decoder=None, debug=False):

        super(PropagationCell, self).__init__(img_size, crop_size, n_what, transition, input_encoder,
                                              glimpse_encoder, transform_estimator, steps_predictor,
                                              decoder=decoder, debug=debug)

        with self._enter_variable_scope():
            self._what_transform = MLP([self._n_hidden] * 2)
            self._latent_scale = latent_scale

    def _parse_inpt(self, inpt, _):
        inpt, _ = inpt
        presence = inpt[2]
        return inpt, presence

    def _compute_what(self, inpt, img, where_code):
        what_tm1 = inpt[0]
        code = self._extract_and_encode_glimpse(img, where_code)

        inpt = tf.concat((code, what_tm1), -1)
        what_params = self._what_transform(inpt)
        what_params *= self._latent_scale

        what_distrib = self._what_distrib(what_params)
        return what_distrib.sample(), what_distrib.loc, what_distrib.scale

    def _compute_where(self, inpt, hidden_output):
        where_tm1 = inpt[1]

        inpt = tf.concat((hidden_output, where_tm1), -1)

        loc, scale = self._transform_estimator(inpt)
        loc *= self._latent_scale
        where_distrib = NormalWithSoftplusScale(loc, scale,
                                                validate_args=self._debug, allow_nan_stats=not self._debug)

        return where_distrib.sample(), where_distrib.loc, where_distrib.scale

    def _compute_presence(self, inpt, presence, hidden_output):
        what_tm1, where_tm1, presence_tm1, presence_logit_tm1 = inpt


        inpt = tf.concat((what_tm1, where_tm1, hidden_output), -1)
        presence_logit = self._steps_predictor(inpt) #+ presence_logit_tm1
        presence_prob = tf.nn.sigmoid(presence_logit)

        presence_distrib = Bernoulli(probs=presence_prob, dtype=tf.float32,
                                     validate_args=self._debug, allow_nan_stats=not self._debug)
        new_presence = presence_distrib.sample()
        # object can be present only if it was present at the previous timestep and it does not depend on different
        # object at this timestep
        presence = presence_tm1 * new_presence

        return presence, presence_prob, presence_logit