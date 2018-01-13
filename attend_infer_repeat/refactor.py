import collections
import sonnet as snt
import tensorflow as tf
from attrdict import AttrDict
from tensorflow.contrib.distributions import Bernoulli, Categorical, Geometric, Normal
from tensorflow.python.util import nest


from elbo import kl_by_sampling
from ops import (stack_states, extract_state, clip_preserve, update_num_obj_counts, compute_object_ids,
                 select_present_list)
from prior import PoissonBinomialDistribution, NumStepsDistribution
from neural import MLP


_latent_name_to_idx = dict(
    what=0,
    where=3,
    pres_prob=6,
    pres=7,
    pres_logit=8,
)


def _extract_latents(hidden_outputs, key=None, skip=None):
    if key is not None and skip is not None:
        raise ValueError("Either `key' or `skip' have to be None, but both are not!")

    if skip is not None:
        key = (k for k in _latent_name_to_idx.keys() if k not in nest.flatten(skip))
        latent_idx = sorted((_latent_name_to_idx[k] for k in key))

    elif key is None:
        latent_idx = sorted(_latent_name_to_idx.values())
    else:
        latent_idx = (_latent_name_to_idx[k] for k in nest.flatten(key))

    if isinstance(hidden_outputs[0], tf.Tensor):
        latents = [hidden_outputs[i] for i in latent_idx]
    else:
        latents = [extract_state(hidden_outputs, i) for i in latent_idx]

    if len(latents) == 1:
        latents = latents[0]
    return latents


class AIRBase(snt.AbstractModule):
    def __init__(self, num_step_distribution_class, n_steps, batch_size, cell):
        super(AIRBase, self).__init__()
        self._n_steps = n_steps
        self._batch_size = batch_size
        self._cell = cell
        self._num_step_distribution_class = num_step_distribution_class

    def _unroll_timestep(self, inpt, hidden_state, seq_len=None):

        if seq_len is None:
            seq_len_inpt = [tf.ones((self._batch_size, 1))] * self._n_steps
        else:
            seq_len_inpt = []
            for t in xrange(self._n_steps):
                exists = tf.greater(seq_len, t)
                seq_len_inpt.append(tf.expand_dims(tf.to_float(exists), -1))

        inpt = [[i, s] for i, s in zip(inpt, seq_len_inpt)]

        hidden_outputs, hidden_state = tf.nn.static_rnn(self._cell, inpt, hidden_state)
        return hidden_outputs, hidden_state[-1]

    def _make_posteriors(self, hidden_outputs):
        what, what_loc, what_scale, where, where_loc, where_scale, presence_prob, presence, presence_logit \
            = hidden_outputs

        num_steps_posterior = self._num_step_distribution_class(presence_prob[..., 0])

        where_posterior = Normal(where_loc, where_scale)
        what_posterior = Normal(what_loc, what_scale)

        return what_posterior, where_posterior, num_steps_posterior

    def _outputs_by_name(self, hidden_outputs):
        return {n: o for n, o in zip(self._cell.output_names, hidden_outputs)}


class AttendDiscoverRepeat(AIRBase):
    def __init__(self, n_steps, batch_size, cell, step_success_prob):
        super(AttendDiscoverRepeat, self).__init__(NumStepsDistribution, n_steps, batch_size, cell)
        self._what_prior = Normal(0., 1.)
        self._where_prior = Normal(0., 1.)
        self._num_steps_prior = Geometric(1. - step_success_prob)

    def _build(self, img, n_present_obj, conditioning=None):
        hidden_outputs, num_steps, log_pres_prob = self._discover(img, n_present_obj, conditioning)
        hidden_outputs = stack_states(hidden_outputs)
        kl, kl_where, kl_what, kl_num_step = self._estimate_kl(hidden_outputs, num_steps)

        outputs = AttrDict(
            hidden_outputs=hidden_outputs,
            num_steps=num_steps,
            log_pres_prob=log_pres_prob,
            kl=kl,
            kl_where=kl_where,
            kl_what=kl_what,
            kl_num_step=kl_num_step
        )
        outputs.update(self._outputs_by_name(hidden_outputs))

        return outputs

    def _discover(self, img, n_present_obj, conditioning):
        max_disc_steps = self._n_steps - n_present_obj
        initial_state = self._cell.initial_state(img)

        if conditioning is None:
            conditioning = tf.zeros((self._batch_size, 1))

        conditioning = [conditioning] * self._n_steps
        hidden_outputs, inner_hidden_state = self._unroll_timestep(conditioning, initial_state,
                                                                   seq_len=max_disc_steps)
        pres_prob, presence = _extract_latents(hidden_outputs, key='pres_prob pres'.split())
        num_steps = tf.reduce_sum(presence[..., 0], -1)
        log_pres_prob = NumStepsDistribution(pres_prob[..., 0]).log_prob(num_steps[..., tf.newaxis])

        return hidden_outputs, num_steps, log_pres_prob

    def _estimate_kl(self, hidden_outputs, num_steps):
        what, what_loc, what_scale, where, where_loc, where_scale, presence_prob, presence, presence_logit \
            = hidden_outputs

        presence = presence[..., 0]

        # Latent Distribs
        what_posterior, where_posterior, num_steps_posterior = self._make_posteriors(hidden_outputs)

        # KLs
        kl_what = kl_by_sampling(what_posterior, self._what_prior, what)
        kl_what = tf.reduce_sum(kl_what, -1) * presence

        kl_where = kl_by_sampling(where_posterior, self._where_prior, where)
        kl_where = tf.reduce_sum(kl_where, -1) * presence

        kl_num_step = kl_by_sampling(num_steps_posterior, self._num_steps_prior, num_steps)
        kl = tf.reduce_sum(kl_what + kl_where, -1) + kl_num_step

        return kl, kl_where, kl_what, kl_num_step


class AttendPropagateRepeat(AIRBase):
    _num_step_distribution_class = PoissonBinomialDistribution

    def __init__(self, n_steps, batch_size, cell, prior_cell):
        super(AttendPropagateRepeat, self).__init__(PoissonBinomialDistribution, n_steps, batch_size, cell)
        self._prior_cell = prior_cell

    @property
    def n_what(self):
        return self._cell._n_what

    def prior_init_state(self, trainable=True, initializer=None):
        if initializer is not None and not isinstance(initializer, collections.Sequence):
            state_size = self._prior_cell.state_size
            flat_state_size = nest.flatten(state_size)
            initializer = [initializer] * len(flat_state_size)
            initializer = nest.pack_sequence_as(state_size, initializer)

        init_state = self._prior_cell.initial_state(self._batch_size, tf.float32,
                                                    trainable=trainable,
                                                    trainable_initializers=initializer)
        return init_state

    def _build(self, img, z_tm1, temporal_state, prior_state):
        presence_tm1 = z_tm1[2]

        prior_stats, prior_state = self._update_prior(z_tm1, prior_state)
        hidden_outputs, num_steps, log_pres_prob = self._propagate(img, z_tm1, temporal_state)
        hidden_outputs = stack_states(hidden_outputs)
        kl, kl_where, kl_what, kl_num_step = self._estimate_kl(presence_tm1, hidden_outputs, num_steps, prior_stats)

        outputs = AttrDict(
            prior_stats=prior_stats,
            prior_state=prior_state,
            hidden_outputs=hidden_outputs,
            num_steps=num_steps,
            log_pres_prob=log_pres_prob,
            kl=kl,
            kl_where=kl_where,
            kl_what=kl_what,
            kl_num_step=kl_num_step
        )
        outputs.update(self._outputs_by_name(hidden_outputs))

        return outputs

    def _temporal_to_step_hidden_state(self, temporal_hidden_state):
        """Linear projection of the temporal hidden state to the step-wise hidden state"""
        flat_hidden_state = tf.concat(nest.flatten(temporal_hidden_state), -1)
        state_size = self._cell.state_size[-1]
        flat_state_size = sum([int(s) for s in state_size])
        state = snt.Linear(flat_state_size)(flat_hidden_state)
        state = tf.split(state, state_size, -1)
        if len(state) == 1:
            state = state[0]

        return state

    def _propagate(self, img, z_tm1, temporal_hidden_state):
        # # 1) propagate previous
        step_initial_state = self._temporal_to_step_hidden_state(temporal_hidden_state)
        initial_state = self._cell.initial_state(img, hidden_state=step_initial_state)

        unstacked_z_tm1 = zip(*[[zz[:, 0] for zz in tf.split(z, self._n_steps, -2)] for z in z_tm1])
        unstacked_z_tm1 = [list(z) for z in unstacked_z_tm1]

        hidden_outputs, inner_hidden_state = self._unroll_timestep(unstacked_z_tm1, initial_state)

        pres_prob, presence = _extract_latents(hidden_outputs, key='pres_prob pres'.split())
        num_steps = tf.reduce_sum(presence[..., 0], -1)
        pres_tm1 = z_tm1[2]

        clipped_prob = clip_preserve(pres_prob, 1e-16, 1. - 1e-7)
        # # this is only for gradient computation later
        # # if there was no object at the previous timestep, there's zero prob of being it now
        # # so we just multiply by the presence before to mask out non-existent objects

        log_pres_prob = Bernoulli(clipped_prob).log_prob(presence) * pres_tm1
        log_pres_prob = tf.reduce_sum(log_pres_prob, -2)[..., 0]

        return hidden_outputs, num_steps, log_pres_prob

    def _update_prior(self, z_tm1, prior_hidden_state):
        what_tm1, where_tm1, presence_tm1 = z_tm1[:3]

        rnn_hidden_state, num_obj_counts = prior_hidden_state

        # continuous
        prior_rnn_inpt = tf.concat((what_tm1, where_tm1), -1)
        rnn = snt.BatchApply(self._prior_cell)

        outputs, rnn_hidden_state = rnn(prior_rnn_inpt, rnn_hidden_state)
        stats = snt.BatchApply(snt.Linear(2 * (4 + self.n_what)))(outputs)

        locs, scales = tf.split(stats, 2, -1)
        prior_where_loc, prior_what_loc = tf.split(locs, [4, self.n_what], -1)
        prior_where_scale, prior_what_scale = tf.split(tf.nn.softplus(scales), [4, self.n_what], -1)

        prior_stats = [prior_where_loc, prior_where_scale, prior_what_loc, prior_what_scale]

        # discrete
        num_obj_tm1 = tf.reduce_sum(presence_tm1[..., 0], -1)
        num_obj_counts = update_num_obj_counts(num_obj_counts, num_obj_tm1)

        num_obj_prior_probs = tf.to_float(num_obj_counts)
        num_obj_prior_probs = num_obj_prior_probs / tf.reduce_sum(num_obj_prior_probs, -1, keep_dims=True)

        # prepare outputs
        prior_stats.append(num_obj_prior_probs)
        prior_hidden_state = [rnn_hidden_state, num_obj_counts]
        return prior_stats, prior_hidden_state

    def _estimate_kl(self, presence_tm1, hidden_outputs, num_steps, prior_stats):
        what, what_loc, what_scale, where, where_loc, where_scale, presence_prob, presence, presence_logit \
            = hidden_outputs

        presence_tm1 = presence_tm1[..., 0]

        # Latent Distribs
        what_posterior, where_posterior, num_steps_posterior = self._make_posteriors(hidden_outputs)

        # Prior Distribs
        what_prior, where_prior, num_steps_prior = self._make_priors(prior_stats)

        # KLs
        kl_what = kl_by_sampling(what_posterior, what_prior, what)
        kl_what = tf.reduce_sum(kl_what, -1) * presence_tm1

        kl_where = kl_by_sampling(where_posterior, where_prior, where)
        kl_where = tf.reduce_sum(kl_where, -1) * presence_tm1

        kl_num_step = kl_by_sampling(num_steps_posterior, num_steps_prior, num_steps)
        kl = tf.reduce_sum(kl_what + kl_where, -1) + kl_num_step

        return kl, kl_where, kl_what, kl_num_step

    def _make_priors(self, (prior_where_loc, prior_where_scale, prior_what_loc, prior_what_scale, num_obj_prior_probs)):
        what_prior = Normal(prior_what_loc, prior_what_scale)
        where_prior = Normal(prior_where_loc, prior_where_scale)
        num_step_prior = Categorical(probs=num_obj_prior_probs)
        return what_prior, where_prior, num_step_prior


class APDR(snt.AbstractModule):
    def __init__(self, n_steps, batch_size, propagate, discover, time_cell):
        super(APDR, self).__init__()
        self._n_steps = n_steps
        self._batch_size = batch_size
        self._propagate = propagate
        self._discover = discover
        self._time_cell = time_cell

        with self._enter_variable_scope():
            n_units = nest.flatten(self._time_cell.state_size)[0]
            if isinstance(n_units, tf.TensorShape):
                n_units = n_units[0]

            n_units = int(n_units)
            self._latent_encoder = MLP([n_units] * 2)

    def prior_init_state(self, trainable=True, initializer=tf.truncated_normal_initializer(stddev=1e-2)):
        if not hasattr(self, '_prior_init_state'):
            self._prior_init_state = self._propagate.prior_init_state(trainable, initializer)

        return self._prior_init_state

    def tiled_prior_init_state(self, *args, **kwargs):
        if not hasattr(self, '_tiled_prior_init_state'):
            prior_init_state = self.prior_init_state(*args, **kwargs)
            flat = nest.flatten(prior_init_state)
            flat = [tf.tile(tf.expand_dims(f, -2), (1, self._n_steps, 1)) for f in flat]
            self._tiled_prior_init_state = nest.pack_sequence_as(prior_init_state, flat)

        return self._tiled_prior_init_state

    def _build(self, img, z_tm1, temporal_hidden_state, prop_prior_state, highest_used_ids, prev_ids):

        prop_output, disc_output, temporal_hidden_state = \
            self._propagate_and_discover(img, z_tm1, temporal_hidden_state, prop_prior_state)

        hidden_outputs, z_t, obj_ids, prop_prior_state, highest_used_ids = \
            self._choose_latents(prop_output, disc_output, highest_used_ids, prev_ids)

        kl, kl_what, kl_where, kl_num_step = self._estimate_kl(prop_output, disc_output)
        log_pres_prob = prop_output.log_pres_prob + disc_output.log_pres_prob

        outputs = AttrDict(
            hidden_outputs=hidden_outputs,
            obj_ids=obj_ids,
            z_t=z_t,
            prop_prior_state=prop_prior_state,
            ids=obj_ids,
            highest_used_ids=highest_used_ids,
            kl=kl,
            kl_what=kl_what,
            kl_where=kl_where,
            kl_num_step=kl_num_step,
            prop=prop_output,
            disc=disc_output,
            log_pres_prob=log_pres_prob,
            temporal_hidden_state=temporal_hidden_state
        )
        outputs.update(self._discover._outputs_by_name(hidden_outputs))
        outputs['num_steps'] = tf.reduce_sum(outputs.presence[..., 0], -1)

        return outputs

    def _propagate_and_discover(self, img, z_tm1, temporal_hidden_state, prop_prior_state):
        latent_encoding = self._encode_latents(*z_tm1[:-1])
        temporal_conditioning, temporal_hidden_state = self._time_cell(latent_encoding, temporal_hidden_state)

        prop_output = self._propagate(img, z_tm1, temporal_hidden_state, prop_prior_state)

        conditioning_from_prop = self._encode_latents(prop_output.what, prop_output.where, prop_output.presence)
        disc_output = self._discover(img, prop_output.num_steps, conditioning_from_prop)

        return prop_output, disc_output, temporal_hidden_state

    def _choose_latents(self, prop_output, disc_output, highest_used_ids, prev_ids):
        # 3) merge outputs of the two models
        # hidden_outputs = prop_output.hidden_outputs + disc_output.hidden_outputs
        # hidden_outputs = stack_states(hidden_outputs)
        hidden_outputs = [tf.concat((p, d), -2) for p, d in zip(prop_output.hidden_outputs, disc_output.hidden_outputs)]

        # 4) move present states to the beginning, but maintain ordering, e.g. propagated objects
        # should come before the new ones
        highest_used_ids, new_obj_id = compute_object_ids(highest_used_ids, prev_ids,
                                                          prop_output.presence, disc_output.presence)
        presence = _extract_latents(hidden_outputs, 'pres')

        variables_to_partition = list(hidden_outputs)
        variables_to_partition.append(new_obj_id)

        # append prior hidden states to enable re-shuffling them according to presence
        # append init prior rnn hidden state to discovery: discovered objects get fresh hidden state
        # append previous prior rnn hidden state to propagates: these are continued from before
        # prop_and_disc_z_t = self._append_init_z(z_t)
        prop_prior_rnn_state, num_obj_counts = prop_output.prior_state
        prop_and_disc_prior_hidden_states = self._append_init_prop_prior_state(prop_prior_rnn_state)

        # variables_to_partition.extend(prop_and_disc_z_t)
        variables_to_partition.append(prop_and_disc_prior_hidden_states)

        # # merge, partition, split to avoid partitioning each vec separately
        variables_to_partition = select_present_list(variables_to_partition, presence[..., 0], self._batch_size)
        # # keep only self.n_steps latents
        flat_variables_to_partition = nest.flatten(variables_to_partition)
        flat_variables_to_partition = [v[:, :self._n_steps] for v in flat_variables_to_partition]
        variables_to_partition = nest.pack_sequence_as(variables_to_partition, flat_variables_to_partition)

        hidden_outputs, (obj_ids, prop_prior_rnn_state) = variables_to_partition[:-2], variables_to_partition[-2:]
        z_t = _extract_latents(hidden_outputs, skip='pres_prob')
        # z_t = _extract_latents(hidden_outputs)

        prop_prior_state = [prop_prior_rnn_state, num_obj_counts]
        return hidden_outputs, z_t, obj_ids, prop_prior_state, highest_used_ids

    def _estimate_kl(self, prop_output, disc_output):
        kl = prop_output.kl + disc_output.kl
        kl_what = prop_output.kl_what + disc_output.kl_what
        kl_where = prop_output.kl_where + disc_output.kl_where
        kl_num_step = prop_output.kl_num_step + disc_output.kl_num_step
        return kl, kl_what, kl_where, kl_num_step

    def _append_init_prop_prior_state(self, prop_prior_rnn_state):
        init_state = nest.flatten(self.tiled_prior_init_state())
        flat_state = nest.flatten(prop_prior_rnn_state)
        flat_state = [tf.concat((f, i), -2) for f, i in zip(flat_state, init_state)]
        return nest.pack_sequence_as(prop_prior_rnn_state, flat_state)
        # return tf.concat((prop_prior_rnn_state, init_state), -2)

    def _encode_latents(self, what, where, presence):
        inpts = tf.concat((what, where), -1)
        features = snt.BatchApply(self._latent_encoder)(inpts) * presence
        return tf.reduce_sum(features, -2)