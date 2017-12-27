import tensorflow as tf
from tensorflow.contrib.distributions import Normal, Bernoulli, Categorical
from tensorflow.python.util import nest

import sonnet as snt

from elbo import kl_by_sampling, estimate_importance_weighted_elbo
from ops import stack_states, compute_object_ids, clip_preserve, update_num_obj_counts
from prior import NumStepsDistribution, PoissonBinomialDistribution
from cell import AIRCell, PropagatingAIRCell


from ops import select_present_list


def extract_state(states, idx):
    state = [s[idx] for s in states]
    return tf.stack(state, 1)


class NaiveSeqAirMixin(object):

    def _make_cells(self, transition, input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                    **cell_kwargs):

        air_cell = AIRCell(self.img_size, self.glimpse_size, self.n_what, transition,
                           input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                           condition_on_inpt=self.condition_on_prev,
                           debug=self.debug,
                           **cell_kwargs)

        return [air_cell]

    def _initial_state(self):
        what = tf.zeros((1, 1, self.cell._n_what))
        where = tf.zeros((1, 1, self.cell._n_transform_param))
        presence = tf.zeros((1, 1, 1))
        presence_logit = tf.zeros((1, 1, 1))
        z0 = [tf.tile(i, (self.effective_batch_size, self.max_steps, 1)) for i in what, where, presence, presence_logit]
        self.z0 = z0
        initial_state = z0, self.cell.initial_state(self.used_obs[0])[1:]

        time_state = 0.
        if self.time_transition_class is not None:
            self.time_transition = self.time_transition_class(self.cell._n_hidden)
            time_state = self.time_transition.initial_state(self.effective_batch_size, tf.float32, trainable=True)

        tas = []

        def make_ta(shape=[], usual_shape=True):
            if usual_shape:
                shape = [self.batch_size * self.iw_samples] + shape
            ta = tf.TensorArray(tf.float32, self.n_timesteps, dynamic_size=False, element_shape=shape)
            tas.append(ta)
            return ta

        # TODO: prettify
        what_ta = make_ta([self.max_steps, self.n_what])
        what_ta = make_ta([self.max_steps, self.n_what])
        what_ta = make_ta([self.max_steps, self.n_what])
        where_ta = make_ta([self.max_steps, 4])
        where_ta = make_ta([self.max_steps, 4])
        where_ta = make_ta([self.max_steps, 4])
        pres_prob_ta = make_ta([self.max_steps, 1])
        pres_ta = make_ta([self.max_steps, 1])
        pres_logit_ta = make_ta([self.max_steps, 1])
        obj_id_ta = make_ta([self.max_steps, 1])
        step_log_prob_ta = make_ta([1])
        canvas_ta = make_ta(list(self.img_size))
        glimpse_ta = make_ta([self.max_steps] + list(self.glimpse_size))
        posterior_step_prob_ta = make_ta([self.max_steps + 1])
        likelihood_ta = make_ta()
        kl_what_ta = make_ta([self.max_steps])
        kl_where_ta = make_ta([self.max_steps])
        kl_disc_steps_ta = make_ta()
        kl_prop_steps_ta = make_ta()
        kl_steps_ta = make_ta()
        kl_ta = make_ta()
        elbo_ta = make_ta()
        num_prop_step_ta = make_ta()
        num_disc_step_ta = make_ta()
        num_step_ta = make_ta()
        importance_weight_ta = make_ta()
        iw_elbo_ta = make_ta([self.batch_size], False)
        prior_where_loc_ta = make_ta([self.max_steps, 4])
        prior_where_scale_ta = make_ta([self.max_steps, 4])
        prior_what_loc_ta = make_ta([self.max_steps, self.n_what])
        prior_what_scale_ta = make_ta([self.max_steps, self.n_what])
        prior_prop_steps_ta = make_ta([self.max_steps + 1])

        t = tf.constant(0, dtype=tf.int32, name='time')

        cumulative_imp_weights = tf.ones((self.batch_size, self.iw_samples), dtype=tf.float32)
        last_used_id = -tf.ones((self.effective_batch_size, 1))
        prev_ids = -tf.ones((self.effective_batch_size, self.max_steps, 1))
        # TODO: Implement!
        prior_hidden_state = 0
        if self.prior_rnn_class is not None:
            self.prior_rnn = self.prior_rnn_class(self.cell._n_hidden)
            prior_hidden_state = self.prior_rnn.initial_state(self.effective_batch_size, tf.float32, trainable=True)
            prior_hidden_state = tf.tile(prior_hidden_state[:, tf.newaxis], (1, self.max_steps, 1))
            self.init_prior_hidden_state = prior_hidden_state

        obj_num_counts = tf.ones((self.effective_batch_size, self.max_steps + 1), name='obj_num_counts')

        loop_vars = [t, self.used_obs, initial_state, time_state, cumulative_imp_weights, prev_ids,
                     last_used_id, prior_hidden_state, obj_num_counts] + tas

        return loop_vars

    def _loop_cond(self, t, img_seq, *args):
        return t < tf.shape(img_seq)[0]

    def _loop_body(self, t, img_seq, state, time_state, cumulative_imp_weights, prev_ids, last_used_id,
                   prior_hidden_state, obj_num_counts, *tas):

        # parse inputs
        img = img_seq[t]

        z_tm1, old_hidden_state = self._unpack_state(img, state)
        # inner RNN loop
        # TODO: state propagation in the outer loop should be done by a gating RNN
        # Split the inner loop in two parts
        #     a) propagate past objects based on the state and latents
        #      * objects here can only disappear, not appear
        #      * propagate objects ids and prior hidden states from the past
        #     b) discover new objects based on the state
        #      * those two should use different rnns and results should be rearranged (pushed to the bottom of a vector)
        #      * introduce new object ids and init prior hidden state to the initial one
        #
        hidden_outputs, state, time_state, current_ids, last_used_id, prior_hidden_state, z_tm1,\
        disc_num_steps_per_sample, prop_num_steps_per_sample, disc_presence_prob, prop_presence_prob \
            = self._propagate(z_tm1, old_hidden_state, time_state, prev_ids, last_used_id, prior_hidden_state)

        # updates prior_rnn hidden state and prior statistics
        prior_hidden_state, prior_stats = self._propagate_prior(z_tm1, prior_hidden_state)

        # extract latents
        what, where, presence = (hidden_outputs[i] for i in (0, 3, 7))

        # ## Decode
        canvas, glimpse = self.decoder(what, where, presence)
        canvas *= self.output_multiplier

        likelihood, elbo, kl, kl_what, kl_where, kl_prop_steps, kl_disc_steps, kl_steps, num_step_per_sample,\
        posterior_step_probs, prop_num_step_prior_probs \
            = self._compute_elbo(img, canvas, hidden_outputs, obj_num_counts, disc_num_steps_per_sample,
                                 prop_num_steps_per_sample, disc_presence_prob, prop_presence_prob, *prior_stats)

        obj_num_counts = update_num_obj_counts(obj_num_counts, num_step_per_sample)

        ### importance_weights
        iw_elbo, importance_weights = estimate_importance_weighted_elbo(self.batch_size, self.iw_samples, elbo)
        cumulative_imp_weights *= importance_weights
        cumulative_imp_weights /= tf.reduce_sum(cumulative_imp_weights, -1, keep_dims=True) + 1e-8

        flat_iw = tf.reshape(importance_weights, (self.batch_size * self.iw_samples,))

        # write outputs
        tas = list(tas)
        outputs = hidden_outputs + [canvas, glimpse, posterior_step_probs, likelihood, kl_what, kl_where, kl_prop_steps,
                                    kl_disc_steps, kl_steps, kl, elbo, disc_num_steps_per_sample,
                                    prop_num_steps_per_sample, num_step_per_sample, flat_iw, iw_elbo] + prior_stats + [prop_num_step_prior_probs]
        for i, (ta, output) in enumerate(zip(tas, outputs)):
            tas[i] = ta.write(t, output)

        # Increment time index
        t += 1
        return [t, img_seq, state, time_state, cumulative_imp_weights, current_ids, last_used_id,
                prior_hidden_state, obj_num_counts] + tas

    def _unpack_state(self, img, state):
        """Takes the img and the hidden state from the previous timestep and prepare the hidden state
        for the inner rnn"""
        batch_size = int(img.shape[0])
        z_tm1, inner_state = state

        # prepare latents
        z_tm1 = [tf.reshape(i, (batch_size, self.max_steps, -1)) for i in z_tm1]

        # prepare state
        flat_img = tf.reshape(img, (batch_size, self.cell._n_pix))
        hidden_state = [flat_img] + inner_state
        return z_tm1, hidden_state

    def _pack_state(self, hidden_outputs, old_hidden_state, inner_rnn_state):
        """Takes outputs of the inner RNN and prepares a hidden state to propagate it to the next timestep"""
        zs = [hidden_outputs[i] for i in (0, 3, 7, 8)]  # what, where, presence
        inner_state = old_hidden_state[1:]
        inner_state[-1] = inner_rnn_state  # update the inner RNN state but don't touch initial values for zs
        return zs, inner_state

    def _extract_prev_latents(self, z_tm1):
        prev_latents = []
        for i in xrange(self.max_steps):
            z_tm1_per_object = [z[..., i, :] for z in z_tm1]
            # mask out absent objects
            pres = z_tm1_per_object[2]
            z_tm1_per_object = [z * pres for z in z_tm1_per_object]
            prev_latents.append(z_tm1_per_object)
        return prev_latents

    # def _unroll_timestep(self, inpt, hidden_state, cell, seq_len=None):
    #     if inpt is None:
    #         inpt = [tf.ones((self.effective_batch_size, 1))] * self.max_steps
    #
    #     hidden_outputs, hidden_state = tf.nn.static_rnn(cell, inpt, hidden_state, sequence_length=seq_len)
    #     return hidden_outputs, hidden_state[-1]

    def _unroll_timestep(self, inpt, hidden_state, cell, seq_len=None):
        if inpt is None:
            if seq_len is None:
                inpt = [tf.ones((self.effective_batch_size, 1))] * self.max_steps
            else:
                inpt = []
                for t in xrange(self.max_steps):
                    exists = tf.greater(seq_len, t)
                    inpt.append(tf.expand_dims(tf.to_float(exists), -1))

        hidden_outputs, hidden_state = tf.nn.static_rnn(cell, inpt, hidden_state)
        return hidden_outputs, hidden_state[-1]

    def _time_transiton(self, inner_rnn_state, time_state):
        if self.time_transition_class is not None:
            flat_rnn_state = nest.flatten(inner_rnn_state)
            inpt = tf.concat(flat_rnn_state, -1)
            flat_rnn_state[-1], time_state = self.time_transition(inpt, time_state)
            inner_rnn_state = nest.pack_sequence_as(inner_rnn_state, flat_rnn_state)
        return inner_rnn_state, time_state

    def _propagate(self, z_tm1, prev_hidden_state, time_state, prev_ids, last_used_id, prior_hidden_state):
        """Computes a single time-step
        """
        prev_latents = self._extract_prev_latents(z_tm1)
        hidden_outputs, inner_rnn_state = self._unroll_timestep(prev_latents, prev_hidden_state, self.cell)

        # handle transition between timesteps by a different RNN
        inner_rnn_state, time_state = self._time_transiton(inner_rnn_state, time_state)

        # merge & flatten states
        hidden_outputs = stack_states(hidden_outputs)

        presence_prob, presence = (extract_state(hidden_outputs, i) for i in (6, 7))
        # disc_num_steps_per_sample = tf.reduce_sum(presence, -1)
        disc_num_steps_per_sample = tf.reduce_sum(presence[..., 0], -1)
        prop_num_steps_per_sample = tf.zeros_like(disc_num_steps_per_sample)
        log_prob = NumStepsDistribution(presence_prob[..., 0]).log_prob(disc_num_steps_per_sample)

        # obj id
        hidden_outputs.append(hidden_outputs[-1])
        hidden_outputs.append(log_prob)

        # here we always discover new objects, so let's set the prior hidden state to the initial one
        prior_hidden_state = self.init_prior_hidden_state

        state = self._pack_state(hidden_outputs, prev_hidden_state, inner_rnn_state)
        return hidden_outputs, state, time_state, prev_ids, last_used_id, prior_hidden_state, z_tm1,\
               disc_num_steps_per_sample, prop_num_steps_per_sample, None, None

    def _propagate_prior(self, z_tm1, prior_hidden_state):

        if self.prior_rnn_class is not None:
            prior_rnn_inpt = tf.concat(z_tm1, -1)
            rnn = snt.BatchApply(self.prior_rnn)
            outputs, prior_hidden_state = rnn(prior_rnn_inpt, prior_hidden_state)
            stats = snt.BatchApply(snt.Linear(2 * (4 + self.n_what)))(outputs)
            prior_where_loc, prior_where_scale = tf.split(stats[..., :8], 2, -1)
            prior_what_loc, prior_what_scale = tf.split(stats[..., 8:], 2, -1)

            prior_what_scale, prior_where_scale = (tf.nn.softplus(i) for i in (prior_what_scale, prior_where_scale))
        else:
            o = tf.ones((self.effective_batch_size, self.max_steps, 4))
            prior_where_loc, prior_where_scale = o * 0., o
            o = tf.ones((self.effective_batch_size, self.max_steps, self.n_what))
            prior_what_loc, prior_what_scale = o * 0., o

        return prior_hidden_state, [prior_where_loc, prior_where_scale, prior_what_loc, prior_what_scale]

    def _compute_elbo(self, img, canvas, hidden_outputs, obj_num_counts, disc_num_steps_per_sample,
                                 prop_num_steps_per_sample, disc_presence_prob, prop_presence_prob,
                      prior_where_loc, prior_where_scale, prior_what_loc, prior_what_scale):

        what, what_loc, what_scale, where, where_loc, where_scale, presence_prob, presence, presence_logit\
            = hidden_outputs[:-2]

        ## Output Distribs
        likelihood_per_pixel = Normal(canvas, self.output_std).log_prob(img)
        likelihood = tf.reduce_sum(likelihood_per_pixel, (-2, -1))

        disc_num_steps_posterior = NumStepsDistribution(disc_presence_prob[..., 0])
        posterior_step_probs = disc_num_steps_posterior.prob()
        prop_num_steps_posterior = PoissonBinomialDistribution(prop_presence_prob[..., 0])

        ax = where_loc.shape.ndims - 1
        us, ut = tf.split(where_loc, 2, ax)
        ss, st = tf.split(where_scale, 2, ax)
        scale_posterior = Normal(us, ss)
        shift_posterior = Normal(ut, st)
        what_posterior = Normal(what_loc, what_scale)

        ## KLs
        ordered_step_prob = presence[..., 0]

        what_prior = Normal(prior_what_loc, prior_what_scale)
        scale_loc, shift_loc = tf.split(prior_where_loc, 2, -1)
        scale_scale, shift_scale = tf.split(prior_where_scale, 2, -1)
        scale_prior = Normal(scale_loc, scale_scale)
        shift_prior = Normal(shift_loc, shift_scale)

        ### KL what
        what_kl = kl_by_sampling(what_posterior, what_prior, what)
        kl_what = tf.reduce_sum(what_kl, -1) * ordered_step_prob

        ### KL where
        ax = where.shape.ndims - 1
        scale, shift = tf.split(where, 2, ax)
        scale_kl = kl_by_sampling(scale_posterior, scale_prior, scale)
        shift_kl = kl_by_sampling(shift_posterior, shift_prior, shift)

        scale_kl, shift_kl = [tf.reduce_sum(i * ordered_step_prob[..., tf.newaxis], -1) for i in
                              (scale_kl, shift_kl)]
        kl_where = scale_kl + shift_kl

        ### KL steps
        num_step_per_sample = tf.to_float(tf.reduce_sum(tf.squeeze(presence), -1))
        # kl_steps = kl_by_sampling(disc_num_steps_posterior, self.num_step_prior, num_step_per_sample)

        kl_disc_steps = kl_by_sampling(disc_num_steps_posterior, self.num_step_prior, disc_num_steps_per_sample)

        obj_num_counts = tf.to_float(obj_num_counts)
        prop_num_step_prior_probs = obj_num_counts / tf.reduce_sum(obj_num_counts, -1, keep_dims=True)
        prop_num_step_prior = Categorical(probs=prop_num_step_prior_probs)

        kl_prop_steps = kl_by_sampling(prop_num_steps_posterior, prop_num_step_prior, prop_num_steps_per_sample)
        kl_steps = kl_disc_steps + kl_prop_steps

        ### KL
        kl = tf.reduce_sum(kl_what + kl_where, -1) + kl_steps
        ### elbo
        elbo = likelihood - kl

        return likelihood, elbo, kl, kl_what, kl_where, kl_disc_steps, kl_steps, kl_prop_steps, \
               num_step_per_sample, posterior_step_probs, prop_num_step_prior_probs

    def _time_loop(self):
        """Unrolls the model in time
        """
        loop_vars = self._initial_state()
        res = tf.while_loop(self._loop_cond, self._loop_body, loop_vars, parallel_iterations=self.batch_size)
        return res


class SeparateSeqAIRMixin(NaiveSeqAirMixin):

    def _make_cells(self, transition, input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                    **cell_kwargs):

        discovery_cell = AIRCell(self.img_size, self.glimpse_size, self.n_what, transition,
                           input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                           condition_on_inpt=False,
                           debug=self.debug,
                           **cell_kwargs)

        # Prop cell should have a different rnn cell but should share all other estimators
        prop_transition = transition.__class__(discovery_cell._n_hidden)
        input_encoder = lambda: discovery_cell._input_encoder
        glimpse_encoder = lambda: discovery_cell._glimpse_encoder
        # transform_estimator = lambda: discovery_cell._transform_estimator
        # steps_predictor = lambda: discovery_cell._steps_predictor

        propagation_cell = PropagatingAIRCell(self.img_size, self.glimpse_size, self.n_what, prop_transition,
                           input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                           debug=self.debug)

        return [discovery_cell, propagation_cell]

    def _propagate(self, z_tm1, prev_hidden_state, time_state, prev_ids, last_used_id, prior_hidden_state):
        """Computes a single time-step
        """
        discovery_cell, propagation_cell = self.cells
        prev_latents = self._extract_prev_latents(z_tm1)

        # # 1) propagate previous
        propagate_outputs, inner_prop_state = self._unroll_timestep(prev_latents, prev_hidden_state, propagation_cell)
        prev_hidden_state[-1] = inner_prop_state

        prop_presence_prob, prop_pres = (extract_state(propagate_outputs, i) for i in (6, 7))
        prop_num_steps_per_sample = tf.reduce_sum(prop_pres[..., 0], -1)
        pres_tm1 = z_tm1[2]
        clipped_prob = clip_preserve(prop_presence_prob, 1e-16, 1. - 1e-7)
        # this is only for gradient computation later
        # if there was no object at the previous timestep, there's zero prob of being it now
        # so we just multiply by the presence before to mask out non-existent objects
        prop_log_prob = Bernoulli(clipped_prob).log_prob(prop_pres) * pres_tm1
        prop_log_prob = tf.reduce_sum(prop_log_prob, -2)

        # # 2) discover new objects
        max_disc_steps = self.max_steps - prop_num_steps_per_sample
        discovery_outputs, inner_discovery_state = self._unroll_timestep(None, prev_hidden_state, discovery_cell,
                                                                         seq_len=max_disc_steps)
        prev_hidden_state[-1] = inner_discovery_state

        disc_presence_prob, disc_pres = (extract_state(discovery_outputs, i) for i in (6, 7))
        disc_num_steps_per_sample = tf.reduce_sum(disc_pres[..., 0], -1)
        discovery_log_prob = NumStepsDistribution(disc_presence_prob[..., 0]).log_prob(disc_num_steps_per_sample)
        step_log_prob = prop_log_prob + discovery_log_prob[..., tf.newaxis]

        last_used_id, new_obj_id = compute_object_ids(last_used_id, prev_ids, prop_pres, disc_pres)

        # 3) merge outputs of the two models
        hidden_outputs = propagate_outputs + discovery_outputs

        hidden_outputs = stack_states(hidden_outputs)
        # 4) move present states to the beginning, but maintain ordering, e.g. propagated objects
        # should come before the new ones
        presence = hidden_outputs[7]
        hidden_outputs.append(new_obj_id)

        # append prior hidden states to enable re-shuffling them according to presence
        # append init prior rnn hidden state to discovery: discovered objects get fresh hidden state
        # append previous prior rnn hidden state to propagates: these are continued from before
        if self.prior_rnn_class is not None:
            both_z_tm1 = [tf.concat((a, b), -2) for a, b in zip(z_tm1, self.z0)]
            both_prior_hidden_states = tf.concat((prior_hidden_state, self.init_prior_hidden_state), -2)
            hidden_outputs.extend(both_z_tm1)
            hidden_outputs.append(both_prior_hidden_states)

        # # merge, partition, split to avoid partitioning each vec separately
        hidden_outputs = select_present_list(hidden_outputs, presence[..., 0], self.effective_batch_size)
        # # keep only self.max_steps latents
        hidden_outputs = [ho[:, :self.max_steps] for ho in hidden_outputs]
        if self.prior_rnn_class is not None:
            z_len = len(z_tm1)
            idx = -1 - z_len
            z_tm1, prior_hidden_state, hidden_outputs = hidden_outputs[idx:-1], hidden_outputs[-1], hidden_outputs[:idx]

        # reset ids of forgotten objects to -1
        obj_ids = hidden_outputs[-1]
        hidden_outputs.append(step_log_prob)

        # handle transition between timesteps by a different RNN
        inner_rnn_state, time_state = self._time_transiton(inner_discovery_state, time_state)

        state = self._pack_state(hidden_outputs, prev_hidden_state, inner_rnn_state)
        return hidden_outputs, state, time_state, obj_ids, last_used_id, prior_hidden_state, z_tm1,\
               disc_num_steps_per_sample, prop_num_steps_per_sample, disc_presence_prob, prop_presence_prob