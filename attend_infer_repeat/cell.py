import numpy as  np
import sonnet as snt
import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli, NormalWithSoftplusScale
from tensorflow.python.util import nest

from modules import SpatialTransformer, ParametrisedGaussian


def conditional_concat(cond, *inpts):
    if cond:
        return tf.concat(inpts, -1)
    return inpts[0]


class AIRCell(snt.RNNCore):
    """RNN cell that implements the core features of Attend, Infer, Repeat, as described here:
    https://arxiv.org/abs/1603.08575
    """
    _n_transform_param = 4
    _init_presence_value = 1.  # at the beginning we assume all objects were present

    def __init__(self, img_size, crop_size, n_what,
                 transition, input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                 condition_on_latents=False,
                 condition_on_inpt=False, transition_only_on_object=False, debug=False):
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

        super(AIRCell, self).__init__()
        self._img_size = img_size
        self._n_pix = np.prod(self._img_size)
        self._crop_size = crop_size
        self._n_what = n_what
        self._transition = transition
        self._n_hidden = self._transition.output_size[0]

        self._condition_on_latents = condition_on_latents
        self._condition_on_inpt = condition_on_inpt
        self._debug = debug
        self._transition_only_on_object = transition_only_on_object

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
        return 'what what_loc what_scale where where_loc where_scale presence_prob presence presence_logit'.split()

    def initial_state(self, img):
        batch_size = img.get_shape().as_list()[0]
        hidden_state = self._transition.initial_state(batch_size, tf.float32, trainable=True)

        where_code = tf.zeros([1, self._n_transform_param], dtype=tf.float32, name='where_init')
        what_code = tf.zeros([1, self._n_what], dtype=tf.float32, name='what_init')

        where_code, what_code = (tf.tile(i, (batch_size, 1)) for i in (where_code, what_code))

        flat_img = tf.reshape(img, (batch_size, self._n_pix))
        init_presence = tf.ones((batch_size, 1), dtype=tf.float32) * self._init_presence_value
        return [flat_img, what_code, where_code, init_presence, hidden_state]

    def _prepare_rnn_inputs(self, inpt, img, what, where, presence):
        transition_inpt = self._input_encoder(img)
        transition_inpt = [transition_inpt]
        if inpt is not None and self._condition_on_inpt:
            transition_inpt += nest.flatten(inpt)

        if self._condition_on_latents:
            transition_inpt += [what, where, presence]

        if len(transition_inpt) > 1:
            transition_inpt = tf.concat(transition_inpt, -1)
        else:
            transition_inpt = transition_inpt[0]

        return transition_inpt

    def _compute_what(self, inpt, img, where_code):
        cropped = self._spatial_transformer(img, logits=where_code)
        flat_crop = snt.BatchFlatten()(cropped)
        what_params = self._glimpse_encoder(flat_crop)
        what_distrib = self._what_distrib(what_params)
        return what_distrib.sample(), what_distrib.loc, what_distrib.scale

    def _compute_where(self, inpt, hidden_output):
        where_param = self._transform_estimator(hidden_output)
        where_distrib = NormalWithSoftplusScale(*where_param,
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

    def _maybe_transition(self, presence, inpt, state, new_state):
        bool_pres = tf.cast(presence, bool)
        new_state = [tf.where(tf.tile(bool_pres, (1, int(s.shape[-1]))), ns, s) for ns, s in zip(new_state, state)]
        new_state[-2] = presence
        return new_state

    def postprocess(self, inpt, output, old_state, new_state):
        """Implements sequence lengths

            input is an indicator vector, where one means that a transition is allowed while 0
            means there should be no transition. All outputs for disallowed transitions are set
            to zero, while the hidden state for the last allowed transition is propagated"""

        allowed = inpt
        for i, o in enumerate(output):
            output[i] = allowed * o

        state = []
        for os, ns in zip(old_state, new_state):
            state.append(allowed * ns + (1 - allowed) * os)

        return output, state

    def _build(self, inpt, state):
        """Input is unused; it's only to force a maximum number of steps"""

        img_flat, what_code, where_code, presence, hidden_state = state

        img_inpt = img_flat
        img = tf.reshape(img_inpt, (-1,) + tuple(self._img_size))

        with tf.variable_scope('rnn_inpt'):
            rnn_inpt = self._prepare_rnn_inputs(inpt, img, what_code, where_code, presence)
            hidden_output, hidden_state = self._transition(rnn_inpt, hidden_state)

        with tf.variable_scope('where'):
            where_code, where_loc, where_scale = self._compute_where(inpt, hidden_output)

        with tf.variable_scope('presence'):
            presence, presence_prob, presence_logit\
                = self._compute_presence(inpt, presence, hidden_output)

        with tf.variable_scope('what'):
            what_code, what_loc, what_scale = self._compute_what(inpt, img, where_code)

        output = [what_code, what_loc, what_scale, where_code, where_loc, where_scale,
                  presence_prob, presence, presence_logit]
        new_state = [img_flat, what_code, where_code, presence, hidden_state]

        if self._transition_only_on_object:
            # if the object is not present, we don't update the state
            new_state = self._maybe_transition(presence, inpt, state, new_state)

        return self.postprocess(inpt, output, state, new_state)


class PropagatingAIRCell(AIRCell):
    _init_presence_value = 0.  # at the beginning we assume no objects

    def __init__(self, img_size, crop_size, n_what,
                 transition, input_encoder, glimpse_encoder, transform_estimator, steps_predictor, debug=False):

        super(PropagatingAIRCell, self).__init__(img_size, crop_size, n_what, transition, input_encoder,
                                                 glimpse_encoder, transform_estimator, steps_predictor,
                                                 condition_on_latents=False, condition_on_inpt=True,
                                                 transition_only_on_object=True,
                                                 debug=debug)

    def _compute_where(self, inpt, hidden_output):
        where_tm1 = inpt[1]

        loc, scale = self._transform_estimator(hidden_output)
        loc += where_tm1
        # loc = where_tm1 + snt.Linear(4)(loc)
        where_distrib = NormalWithSoftplusScale(loc, scale,
                                                validate_args=self._debug, allow_nan_stats=not self._debug)

        return where_distrib.sample(), where_distrib.loc, where_distrib.scale

    def _compute_presence(self, inpt, presence, hidden_output):
        presence_logit_tm1 = inpt[3]
        presence_logit = self._steps_predictor(hidden_output) + presence_logit_tm1
        presence_prob = tf.nn.sigmoid(presence_logit)

        presence_distrib = Bernoulli(probs=presence_prob, dtype=tf.float32,
                                     validate_args=self._debug, allow_nan_stats=not self._debug)
        new_presence = presence_distrib.sample()
        presence_tm1 = inpt[2]
        # object can be present only if it was present at the previous timestep and it does not depend on different
        # object at this timestep
        presence = presence_tm1 * new_presence

        return presence, presence_prob, presence_logit

    def _maybe_transition(self, presence, inpt, state, new_state):
        """Transition only if the object was present before"""
        presence_tm1 = inpt[2]
        bool_pres = tf.cast(presence_tm1, bool)
        new_state = [tf.where(tf.tile(bool_pres, (1, int(s.shape[-1]))), ns, s) for ns, s in zip(new_state, state)]
        new_state[3] = presence
        return new_state

    def postprocess(self, inpt, output, old_state, new_state):
        return output, new_state


class SeqAIRCell(snt.RNNCore):

    def __init__(self, n_steps, air_cell):
        super(SeqAIRCell, self).__init__()
        self._n_steps = n_steps
        self._cell = air_cell

    @property
    def state_size(self):
        return [
            [   # from previous timestep
                self._n_steps * self._cell._n_what,  # what
                self._n_steps * self._cell._n_transform_param,  # where
                self._n_steps * 1,  # presence
            ],
            self._cell.state_size[1:],
          # 1, # per-sample elbo estimate, needed in the state for FIVO: resample at the beginning of every step
          # 1, # per-sample cumulative elbo estimate, needed in the state for IWAE: resample after all steps

        ]

    @property
    def output_size(self):
        os = self._cell.output_size
        os = [self._n_steps * o for o in os]

        # self._n_steps, # per-sample per-step kl_what
        # self._n_steps, # per-sample per-step kl_where
        # self._n_steps, # per-sample per-step rec_loss
        # 1,             # per-sample elbo estimate

        return os

    @property
    def output_names(self):
        cell_output_names = self._cell.output_names
        add_names = 'kl_what_per_sample kl_where_prec_sample rec_loss_per_sample nelbo_per_sample'.split()
        return cell_output_names# + add_names

    def initial_inpt(self, batch_size):
        # TODO: trainable or sampled from a prior
        with self._enter_variable_scope():
            what = tf.zeros((1, self._cell._n_what))
            where = tf.zeros((1, self._cell._n_transform_param))
            presence = tf.zeros((1, 1))
            return [tf.tile(i, (batch_size, self._n_steps)) for i in what, where, presence]

    def initial_state(self, img):
        batch_size = int(img.shape[0])
        z_tm1 = self.initial_inpt(batch_size)

        hidden_state = self._cell.initial_state(img)
        return z_tm1, hidden_state[1:]

    def _build(self, img, state):

        batch_size = int(img.shape[0])
        z_tm1, inner_state = state

        # prepare latents
        z_tm1 = [tf.reshape(i, (batch_size, self._n_steps, -1)) for i in z_tm1]

        # prepare state
        flat_img = tf.reshape(img, (batch_size, self._cell._n_pix))
        hidden_state = [flat_img] + inner_state

        # inner RNN loop
        hidden_outputs = []
        for i in xrange(self._n_steps):
            z_tm1_per_object = [z[..., i, :] for z in z_tm1]
            hidden_output, hidden_state = self._cell(z_tm1_per_object, hidden_state)
            hidden_outputs.append(hidden_output)

        # merge & flatten states
        hidden_outputs = zip(*hidden_outputs)
        for i, ho in enumerate(hidden_outputs):
            hidden_outputs[i] = tf.concat(ho, -1)

        # extract latents
        z = [hidden_outputs[i] for i in [0, 3, 7]]

        # we overwrite only the hidden state of the inner rnn
        inner_state[-1] = hidden_state[-1]
        state = z, inner_state

        return hidden_outputs, state