import tensorflow as tf
from tensorflow.contrib.distributions import Normal

from base import BaseAPDRModel
from cell import BaseAPDRCell, DiscoveryCell, PropagationCell
from elbo import estimate_importance_weighted_elbo
from apdr import AttendPropagateRepeat, AttendDiscoverRepeat, APDR


class APDRModel(BaseAPDRModel):
    prop_logit_bias = 3.
    prop_latent_scale_bias = 1.
    decode_prop = False
    constant_prop_prior = False

    def _build_model(self, transition, input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                    **cell_kwargs):

        self.discovery_cell = DiscoveryCell(self.img_size, self.glimpse_size, self.n_what, transition,
                                       input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                                       debug=self.debug,
                                       **cell_kwargs)

        self.n_hidden = self.discovery_cell._n_hidden
        self.n_what = self.discovery_cell._n_what
        self.n_where = self.discovery_cell._n_transform_param
        self.rnn_output_names = self.discovery_cell.output_names
        self.num_rnn_outputs = len(self.rnn_output_names)

        # Prop cell should have a different rnn cell but should share all other estimators
        prop_transition = transition.__class__(self.n_hidden)
        input_encoder = lambda: self.discovery_cell._input_encoder
        glimpse_encoder = lambda: self.discovery_cell._glimpse_encoder

        self.propagation_cell = PropagationCell(self.img_size, self.glimpse_size, self.n_what, prop_transition,
                                           input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                                           debug=self.debug, latent_scale=.1)

        self.discover = AttendDiscoverRepeat(self.max_steps, self.effective_batch_size, self.discovery_cell,
                                             step_success_prob=self._geom_success_prob())

        self.prior_rnn = self.prior_rnn_class(self.n_hidden)
        self.propagate = AttendPropagateRepeat(self.max_steps, self.effective_batch_size, self.propagation_cell,
                                               self.prior_rnn,
                                               prop_logit_bias=self.prop_logit_bias,
                                               latent_scale_bias=self.prop_latent_scale_bias,
                                               constant_prior=self.constant_prop_prior
        )

        self.time_transition = self.time_transition_class(self.n_hidden)

        decoder = self.decoder if self.decode_prop else None
        self.apdr = APDR(self.max_steps, self.effective_batch_size, self.propagate, self.discover,
                         self.time_transition, decoder)

    def _initial_state(self):
        what = tf.zeros((1, 1, self.n_what))
        where = tf.zeros((1, 1, self.n_where))
        presence = tf.zeros((1, 1, 1))
        presence_logit = tf.zeros((1, 1, 1))
        z0 = [tf.tile(i, (self.effective_batch_size, self.max_steps, 1)) for i in what, where, presence, presence_logit]
        self.z0 = z0

        time_state = self.time_transition.initial_state(self.effective_batch_size, tf.float32, trainable=True)

        tas = []
        self.ta_names = []

        def make_ta(name, shape=[], usual_shape=True):
            if usual_shape:
                shape = [self.batch_size * self.iw_samples] + shape
            ta = tf.TensorArray(tf.float32, self.n_timesteps, dynamic_size=False,
                                element_shape=shape)

            tas.append(ta)
            self.ta_names.append(name)

        # RNN outputs
        make_ta('what', [self.max_steps, self.n_what])
        make_ta('what_loc', [self.max_steps, self.n_what])
        make_ta('what_scale', [self.max_steps, self.n_what])
        make_ta('where', [self.max_steps, 4])
        make_ta('where_loc', [self.max_steps, 4])
        make_ta('where_scale', [self.max_steps, 4])
        make_ta('pres_prob', [self.max_steps, 1])
        make_ta('pres', [self.max_steps, 1])
        make_ta('pres_logit', [self.max_steps, 1])

        # Aux, returned as hidden outputs
        make_ta('obj_id', [self.max_steps, 1])
        make_ta('step_log_prob', [1])

        # Others
        make_ta('canvas', list(self.img_size))
        make_ta('glimpse', [self.max_steps] + list(self.glimpse_size))
        make_ta('likelihood_per_sample', )
        make_ta('kl_what_per_sample', [self.max_steps])
        make_ta('kl_where_per_sample', [self.max_steps])
        make_ta('kl_prop_steps_per_sample', )
        make_ta('kl_disc_steps_per_sample', )
        make_ta('kl_steps_per_sample', )
        make_ta('kl_per_sample', )
        make_ta('elbo_per_sample', )
        make_ta('num_prop_step_per_sample', )
        make_ta('num_disc_step_per_sample', )
        make_ta('num_step_per_sample', )
        make_ta('importance_weight', )
        make_ta('iw_elbo', [self.batch_size], False)
        # make_ta('prior_where_loc', [self.max_steps, 4])
        # make_ta('prior_where_scale', [self.max_steps, 4])
        # make_ta('prior_what_loc', [self.max_steps, self.n_what])
        # make_ta('prior_what_scale', [self.max_steps, self.n_what])
        # make_ta('prior_prop_step_probs', [self.max_steps + 1])
        make_ta('prop_prob', [self.max_steps])
        make_ta('disc_prob', [self.max_steps + 1])

        make_ta('prop_pres', [self.max_steps, 1])
        make_ta('disc_pres', [self.max_steps, 1])

        t = tf.constant(0, dtype=tf.int32, name='time')

        cumulative_imp_weights = tf.ones((self.batch_size, self.iw_samples), dtype=tf.float32)
        last_used_id = -tf.ones((self.effective_batch_size, 1))
        prev_ids = -tf.ones((self.effective_batch_size, self.max_steps, 1))

        self.init_prior_prop_state = self.apdr.tiled_prior_init_state()

        loop_vars = [t, self.used_obs, self.z0, time_state, cumulative_imp_weights, prev_ids,
                     last_used_id, self.init_prior_prop_state]
        self.non_ta_output_len = len(loop_vars)

        return loop_vars + tas

    def _loop_body(self, t, img_seq, z_tm1, time_state, cumulative_imp_weights, prev_ids, last_used_id,
                   prop_prior_hidden_state, *tas):

        # parse inputs
        img = img_seq[t]

        apdr_outputs = self.apdr(img, z_tm1, time_state, prop_prior_hidden_state, last_used_id, prev_ids, t)
        hidden_outputs = apdr_outputs.hidden_outputs
        z_t = [what, where, presence, pres_logit] = BaseAPDRCell.extract_latents(hidden_outputs, skip='pres_prob'.split())

        # ## Decode
        canvas, glimpse = self.decoder(what, where, presence)
        canvas *= self.output_multiplier

        likelihood, elbo = self._compute_elbo(img, canvas, apdr_outputs.kl)

        ### importance_weights
        iw_elbo, importance_weights = estimate_importance_weighted_elbo(self.batch_size, self.iw_samples, elbo)
        cumulative_imp_weights *= importance_weights
        cumulative_imp_weights /= tf.reduce_sum(cumulative_imp_weights, -1, keep_dims=True) + 1e-8

        flat_iw = tf.reshape(importance_weights, (self.batch_size * self.iw_samples,))

        # write outputs
        tas = list(tas)
        outputs = list(hidden_outputs) + [
            apdr_outputs.obj_ids,
            tf.expand_dims(apdr_outputs.log_pres_prob, -1),
            canvas,
            glimpse,
            likelihood,
            apdr_outputs.kl_what,
            apdr_outputs.kl_where,
            apdr_outputs.prop.kl_num_step,
            apdr_outputs.disc.kl_num_step,
            apdr_outputs.kl_num_step,
            apdr_outputs.kl,
            elbo,
            apdr_outputs.prop.num_steps,
            apdr_outputs.disc.num_steps,
            apdr_outputs.num_steps,
            flat_iw,
            iw_elbo,
            apdr_outputs.prop.prop_prob,
            apdr_outputs.disc.num_steps_prob,
            apdr_outputs.prop.presence,
            apdr_outputs.disc.presence,
        ]

        for i, (ta, output) in enumerate(zip(tas, outputs)):
            tas[i] = ta.write(t, output)

        # Increment time index
        t += 1
        time_state = apdr_outputs.temporal_hidden_state
        return [t, img_seq, z_t, time_state, cumulative_imp_weights, apdr_outputs.ids, apdr_outputs.highest_used_ids,
                apdr_outputs.prop_prior_state] + tas

    def _compute_elbo(self, img, canvas, kl):

        likelihood_per_pixel = Normal(canvas, self.output_std).log_prob(img)
        likelihood = tf.reduce_sum(likelihood_per_pixel, (-2, -1))
        elbo = likelihood - kl

        return likelihood, elbo

    def _loop_cond(self, t, img_seq, *args):
        return t < tf.shape(img_seq)[0]

    def _time_loop(self):
        """Unrolls the model in time
        """
        loop_vars = self._initial_state()
        res = tf.while_loop(self._loop_cond, self._loop_body, loop_vars, parallel_iterations=self.batch_size)
        return res