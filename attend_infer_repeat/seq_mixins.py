import tensorflow as tf
from tensorflow.contrib.distributions import Normal
from tensorflow.python.util import nest

from elbo import kl_by_sampling, estimate_importance_weighted_elbo
from ops import stack_states
from prior import NumStepsDistribution


class NaiveSeqAirMixin(object):
    def _initial_state(self):
        what = tf.zeros((1, 1, self.cell._n_what))
        where = tf.zeros((1, 1, self.cell._n_transform_param))
        presence = tf.zeros((1, 1, 1))
        z0 = [tf.tile(i, (self.effective_batch_size, self.max_steps, 1)) for i in what, where, presence]
        initial_state = z0, self.cell.initial_state(self.used_obs[0])[1:]

        rnn_hidden_state = self.cell._transition.zero_state(self.effective_batch_size, tf.float32)
        rnn_hidden_state = stack_states([rnn_hidden_state] * self.max_steps)

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
        pres_ta = make_ta([self.max_steps, 1])
        pres_ta = make_ta([self.max_steps, 1])
        canvas_ta = make_ta(list(self.img_size))
        glimpse_ta = make_ta([self.max_steps] + list(self.glimpse_size))
        posterior_step_prob_ta = make_ta([self.max_steps + 1])
        likelihood_ta = make_ta()
        kl_what_ta = make_ta([self.max_steps])
        kl_where_ta = make_ta([self.max_steps])
        kl_steps_ta = make_ta()
        kl_ta = make_ta()
        elbo_ta = make_ta()
        num_step_ta = make_ta()
        importance_weight_ta = make_ta()
        iw_elbo_ta = make_ta([self.batch_size], False)

        t = tf.constant(0, dtype=tf.int32, name='time')

        cumulative_imp_weights = tf.ones((self.batch_size, self.iw_samples), dtype=tf.float32)
        loop_vars = [t, self.used_obs, initial_state, rnn_hidden_state, time_state, cumulative_imp_weights] + tas
        return loop_vars

    def _loop_cond(self, t, img_seq, *args):
        return t < tf.shape(img_seq)[0]

    def _loop_body(self, t, img_seq, state, rnn_hidden_state, time_state, cumulative_imp_weights, *tas):
        # parse inputs
        img = img_seq[t]

        z_tm1, old_hidden_state = self._unpack_state(img, state)

        # inner RNN loop
        # TODO: state propagation in the outer loop should be done by a gating RNN
        # Split the inner loop in two parts
        #     a) propagate past objects based on the state and latents
        #         objects here can only disappear, not appear
        #     b) discover new objects based on the state
        #     those two should use different rnns and results should be rearranged (pushed to the bottom of a vector)
        hidden_outputs, state, rnn_hidden_state, time_state \
            = self._propagate(z_tm1, old_hidden_state, rnn_hidden_state, time_state)

        # extract latents
        what, what_loc, what_scale, where, where_loc, where_scale, presence_prob, presence = hidden_outputs

        # ## Decode
        canvas, glimpse = self.decoder(what, where, presence)
        canvas *= self.output_multiplier

        likelihood, elbo, kl, kl_what, kl_where, kl_steps, num_step_per_sample, posterior_step_probs \
            = self._compute_elbo(img, canvas, z_tm1, hidden_outputs)

        ### importance_weights
        iw_elbo, importance_weights = estimate_importance_weighted_elbo(self.batch_size, self.iw_samples, elbo)
        cumulative_imp_weights *= importance_weights
        cumulative_imp_weights /= tf.reduce_sum(cumulative_imp_weights, -1, keep_dims=True) + 1e-8

        flat_iw = tf.reshape(importance_weights, (self.batch_size * self.iw_samples,))

        # write outputs
        tas = list(tas)
        outputs = hidden_outputs + [canvas, glimpse, posterior_step_probs, likelihood, kl_what, kl_where, kl_steps, kl,
                                    elbo, num_step_per_sample, flat_iw, iw_elbo]
        for i, (ta, output) in enumerate(zip(tas, outputs)):
            tas[i] = ta.write(t, output)

        # Increment time index
        t += 1
        return [t, img_seq, state, rnn_hidden_state, time_state, cumulative_imp_weights] + tas

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
        zs = [hidden_outputs[i] for i in (0, 3, 7)]  # what, where, presence
        inner_state = old_hidden_state[1:]
        inner_state[-1] = inner_rnn_state  # update the inner RNN state but don't touch initial values for zs
        return zs, inner_state

    def _propagate(self, z_tm1, prev_hidden_state, rnn_hidden_state, time_state):
        """Computes a single time-step
        """
        hidden_state = prev_hidden_state
        hidden_outputs, hidden_states = [], []
        for i in xrange(self.max_steps):
            rnn_inpt = []
            if self.condition_on_prev:
                z_tm1_per_object = [z[..., i, :] for z in z_tm1]
                # mask out absent objects
                pres = z_tm1_per_object[-1]
                z_tm1_per_object = [z * pres for z in z_tm1_per_object]
                rnn_inpt.extend(z_tm1_per_object)
            if self.condition_on_rnn_output:
                rnn_inpt.append(rnn_hidden_state[..., i, :])

            if not rnn_inpt:
                rnn_inpt = None

            hidden_output, hidden_state = self.cell(rnn_inpt, hidden_state)
            hidden_outputs.append(hidden_output)
            hidden_states.append(hidden_state[-1])

        # time transition
        inner_rnn_state = hidden_state[-1]
        # handle transition between timesteps by a different RNN
        if self.time_transition_class is not None:
            flat_rnn_state = nest.flatten(inner_rnn_state)
            inpt = tf.concat(flat_rnn_state, -1)
            flat_rnn_state[-1], time_state = self.time_transition(inpt, time_state)
            inner_rnn_state = nest.pack_sequence_as(inner_rnn_state, flat_rnn_state)

        # merge & flatten states
        hidden_outputs = stack_states(hidden_outputs)
        rnn_hidden_state = stack_states(hidden_states)

        state = self._pack_state(hidden_outputs, prev_hidden_state, inner_rnn_state)
        return hidden_outputs, state, rnn_hidden_state, time_state

    def _compute_elbo(self, img, canvas, z_tm1, hidden_outputs):
        what, what_loc, what_scale, where, where_loc, where_scale, presence_prob, presence = hidden_outputs
        num_step_per_sample = tf.to_float(tf.reduce_sum(tf.squeeze(presence), -1))

        ## Output Distribs
        likelihood_per_pixel = Normal(canvas, self.output_std).log_prob(img)
        likelihood = tf.reduce_sum(likelihood_per_pixel, (-2, -1))

        num_steps_posterior = NumStepsDistribution(presence_prob[..., 0])
        posterior_step_probs = num_steps_posterior.prob()

        ax = where_loc.shape.ndims - 1
        us, ut = tf.split(where_loc, 2, ax)
        ss, st = tf.split(where_scale, 2, ax)
        scale_posterior = Normal(us, ss)
        shift_posterior = Normal(ut, st)
        what_posterior = Normal(what_loc, what_scale)

        ## KLs
        ordered_step_prob = presence[..., 0]

        if self.prior_around_prev:
            # block gradient path
            what_tm1, where_tm1 = [tf.stop_gradient(i) for i in z_tm1[:-1]]
            what_prior = Normal(what_tm1, 1.)

            ax = where_tm1.shape.ndims - 1
            scale_prior_loc, shift_prior_loc = tf.split(where_tm1, 2, ax)
            scale_prior = Normal(scale_prior_loc, 1.)
            shift_prior = Normal(shift_prior_loc, 1.)

        else:
            what_prior = self.what_prior
            scale_prior = self.scale_prior
            shift_prior = self.shift_prior

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
        kl_steps = kl_by_sampling(num_steps_posterior, self.num_step_prior, num_step_per_sample)
        ### KL
        kl = tf.reduce_sum(kl_what + kl_where, -1) + kl_steps
        ### elbo
        elbo = likelihood - kl

        return likelihood, elbo, kl, kl_what, kl_where, kl_steps, num_step_per_sample, posterior_step_probs

    def _time_loop(self):
        """Unrolls the model in time
        """
        loop_vars = self._initial_state()
        res = tf.while_loop(self._loop_cond, self._loop_body, loop_vars, parallel_iterations=self.batch_size)
        return res
