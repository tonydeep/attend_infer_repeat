import functools

import tensorflow as tf
from tensorflow.contrib.distributions import Normal

from cell import AIRCell
from evaluation import gradient_summaries
from prior import NumStepsDistribution
from modules import AIRDecoder
from elbo import kl_by_sampling
from grad import VIMCOEstimator
from ops import tile_input_for_iwae, gather_axis


# TODO: implement FIVO & per-timestep VIMCO for FIVO


class SeqAIRModel(object):
    """Generic AIR model

    :param analytic_kl_expectation: bool, computes expectation over conditional-KL analytically if True
    """

    def __init__(self, obs, max_steps, glimpse_size,
                 n_what, transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
                 steps_predictor,
                 output_std=1., output_multiplier=1., iw_samples=1,
                 debug=False, **cell_kwargs):
        """Creates the model.

        :param obs: tf.Tensor, images
        :param max_steps: int, maximum number of steps to take (or objects in the image)
        :param glimpse_size: tuple of ints, size of the attention glimpse
        :param n_what: int, number of latent variables describing an object
        :param transition: see :class: AIRCell
        :param input_encoder: see :class: AIRCell
        :param glimpse_encoder: see :class: AIRCell
        :param glimpse_decoder: callable, decodes the glimpse from latent representation
        :param transform_estimator: see :class: AIRCell
        :param steps_predictor: see :class: AIRCell
        :param output_std: float, std. dev. of the output Gaussian distribution
        :param output_multiplier: float, a factor that multiplies the reconstructed glimpses
        :param debug: see :class: AIRCell
        :param **cell_kwargs: all other parameters are passed to AIRCell
        """

        self.obs = obs
        self.max_steps = max_steps
        self.glimpse_size = glimpse_size
        self.n_what = n_what
        self.output_std = output_std
        self.output_multiplier = output_multiplier
        self.iw_samples = iw_samples
        self.debug = debug

        shape = self.obs.get_shape().as_list()
        self.n_timesteps = shape[0]
        self.batch_size = shape[1]

        self.img_size = shape[2:]
        self.effective_batch_size = self.batch_size * self.iw_samples
        self.used_obs = tile_input_for_iwae(obs, self.iw_samples, with_time=True)

        with tf.variable_scope(self.__class__.__name__):
            self.output_multiplier = tf.Variable(output_multiplier, dtype=tf.float32, trainable=False, name='canvas_multiplier')

            # save existing variables to know later what we've created
            previous_vars = tf.trainable_variables()

            self._build(transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
                        steps_predictor, cell_kwargs)

            # group variables
            model_vars = set(tf.trainable_variables()) - set(previous_vars)
            self.decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                  scope=self.decoder.variable_scope.name)
            self.encoder_vars = list(model_vars - set(self.decoder_vars))
            self.model_vars = list(model_vars)

    def _build(self, transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
               steps_predictor, cell_kwargs):
        """Build the model. See __init__ for argument description"""

        self.num_step_prior_prob, self.num_step_prior,\
        self.scale_prior, self.shift_prior, self.what_prior = self._make_priors()

        self.decoder = AIRDecoder(self.img_size, self.glimpse_size, glimpse_decoder, batch_dims=2)
        air_cell = AIRCell(self.img_size, self.glimpse_size, self.n_what, transition,
                            input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                            debug=self.debug,
                            **cell_kwargs)

        self.cell = air_cell

        # TODO: extract to a method; those should be sampled from a q and there should be a prior on that
        what = tf.zeros((1, 1, self.cell._n_what))
        where = tf.zeros((1, 1, self.cell._n_transform_param))
        presence = tf.zeros((1, 1, 1))
        z0 = [tf.tile(i, (self.effective_batch_size, self.max_steps, 1)) for i in what, where, presence]
        initial_state = z0, self.cell.initial_state(self.used_obs[0])[1:]

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
        loop_vars = [t, self.used_obs, initial_state, cumulative_imp_weights] + tas
        
        def cond(t, img_seq, *args):
            return t < tf.shape(img_seq)[0]
            
        def body(t, img_seq, state, cumulative_imp_weights, *tas):
            # parse inputs
            img = img_seq[t]

            # #################################################

            batch_size = int(img.shape[0])
            z_tm1, inner_state = state

            # prepare latents
            z_tm1 = [tf.reshape(i, (batch_size, self.max_steps, -1)) for i in z_tm1]

            # prepare state
            flat_img = tf.reshape(img, (batch_size, self.cell._n_pix))
            hidden_state = [flat_img] + inner_state

            # inner RNN loop
            # TODO: state propagation in the outer loop should be done by a gating RNN
            # Split the inner loop in two parts
            #     a) propagate past objects based on the state and latents
            #         objects here can only disappear, not appear
            #     b) discover new objects based on the state
            #     those two should use different rnns and results should be rearranged (pushed to the bottom of a vector)
            hidden_outputs = []
            for i in xrange(self.max_steps):
                z_tm1_per_object = [z[..., i, :] for z in z_tm1]
                hidden_output, hidden_state = self.cell(z_tm1_per_object, hidden_state)
                hidden_outputs.append(hidden_output)

            # merge & flatten states
            hidden_outputs = zip(*hidden_outputs)
            for i, ho in enumerate(hidden_outputs):
                hidden_outputs[i] = tf.stack(ho, 1)

            # extract latents
            what, what_loc, what_scale, where, where_loc, where_scale, presence_prob, presence = hidden_outputs
            num_step_per_sample = tf.to_float(tf.reduce_sum(tf.squeeze(presence), -1))

            # we overwrite only the hidden state of the inner rnn
            inner_state[-1] = hidden_state[-1]
            state = [what, where, presence], inner_state

            # #################################################
            # Compute ELBOs
            ## Decode
            canvas, glimpse = self.decoder(what, where, presence)
            canvas *= self.output_multiplier

            ## Output Distribs
            likelihood_per_pixel = Normal(canvas, self.output_std).log_prob(img)
            likelihood = tf.reduce_sum(likelihood_per_pixel, (-2, -1))

            num_steps_posterior = NumStepsDistribution(tf.squeeze(presence_prob))
            posterior_step_probs = num_steps_posterior.prob()

            ax = where_loc.shape.ndims - 1
            us, ut = tf.split(where_loc, 2, ax)
            ss, st = tf.split(where_scale, 2, ax)
            scale_posterior = Normal(us, ss)
            shift_posterior = Normal(ut, st)
            what_posterior = Normal(what_loc, what_scale)

            ## KLs
            ordered_step_prob = tf.squeeze(presence)

            ### KL what
            what_kl = kl_by_sampling(what_posterior, self.what_prior, what)
            kl_what = tf.reduce_sum(what_kl, -1) * ordered_step_prob

            ### KL where
            ax = where.shape.ndims - 1
            scale, shift = tf.split(where, 2, ax)
            scale_kl = kl_by_sampling(scale_posterior, self.scale_prior, scale)
            shift_kl = kl_by_sampling(shift_posterior, self.shift_prior, shift)

            scale_kl, shift_kl = [tf.reduce_sum(i * ordered_step_prob[..., tf.newaxis], -1) for i in
                                  (scale_kl, shift_kl)]
            kl_where = scale_kl + shift_kl

            ### KL steps
            kl_steps = kl_by_sampling(num_steps_posterior, self.num_step_prior,
                                                         num_step_per_sample)
            ### KL
            kl = tf.reduce_sum(kl_what + kl_where, -1) + kl_steps

            ### elbo
            elbo = likelihood - kl

            ### importance_weights
            iw_elbo, importance_weights = estimate_importance_weighted_elbo(self.batch_size, self.iw_samples, elbo)
            cumulative_imp_weights *= importance_weights
            cumulative_imp_weights /= tf.reduce_sum(cumulative_imp_weights, -1, keep_dims=True) + 1e-8

            flat_iw = tf.reshape(importance_weights, (self.batch_size * self.iw_samples,))

            # write outputs
            tas = list(tas)
            outputs = hidden_outputs + [canvas, glimpse, posterior_step_probs, likelihood, kl_what, kl_where, kl_steps, kl, elbo, num_step_per_sample, flat_iw, iw_elbo]
            for i, (ta, output) in enumerate(zip(tas, outputs)):
                tas[i] = ta.write(t, output)

            # Increment time index
            t += 1
            return [t, img_seq, state, cumulative_imp_weights] + tas
                
        res = tf.while_loop(cond, body, loop_vars, parallel_iterations=self.batch_size)
        self.final_state = res[2]
        self.cumulative_imp_weights = res[3]
        tas = res[4:]

        # TODO: prettify
        self.output_names = 'canvas glimpse posterior_step_prob likelihood_per_sample kl_what_per_sample' \
                            ' kl_where_per_sample kl_steps_per_sample kl_per_sample elbo_per_sample num_step_per_sample' \
                            ' importance_weight iw_elbo'.split()
        for name, ta in zip(self.cell.output_names + self.output_names, tas):
            output = ta.stack()
            setattr(self, name, output)

        self.cumulative_elbo_per_sample = tf.reduce_sum(self.elbo_per_sample, 0)

        # self.cumulative_iw_elbo_per_sample = tf.reduce_sum(self.iw_elbo, 0)
        self.cumulative_iw_elbo_per_sample, _ = estimate_importance_weighted_elbo(self.batch_size, self.iw_samples, self.cumulative_elbo_per_sample)

        resampling_logits = tf.reshape(self.cumulative_elbo_per_sample, (self.batch_size, self.iw_samples))
        self.cumulative_imp_distrib = tf.contrib.distributions.Categorical(resampling_logits)
        self.imp_resampling_idx = self.cumulative_imp_distrib.sample()

        # Logging
        self._log_resampled(-self.likelihood_per_sample, 'rec_loss')
        self._log_resampled(self.kl_per_sample, 'kl_div')
        self._log_resampled(self.num_step_per_sample, 'num_step')
        self._log_resampled(tf.reduce_sum(self.kl_what_per_sample, -1), 'kl_what')
        self._log_resampled(tf.reduce_sum(self.kl_what_per_sample, -1), 'kl_where')
        self._log_resampled(self.kl_steps_per_sample, 'kl_num_steps')

        # For rendering
        resampled_names = 'canvas glimpse presence where posterior_step_prob'.split()
        for name in resampled_names:
            setattr(self, 'resampled_' + name, self.resample(getattr(self, name), axis=1))

    def train_step(self, learning_rate, nums=None,
                   optimizer=tf.train.RMSPropOptimizer, opt_kwargs=dict(momentum=.9, centered=True)):
        """Creates the train step and the global_step

        :param learning_rate: float or tf.Tensor
        :param nums: tf.Tensor, number of objects in images
        :return: train step and global step
        """

        with tf.variable_scope('loss'):
            self.learning_rate = tf.Variable(learning_rate, name='learning_rate', trainable=False)
            make_opt = functools.partial(optimizer, **opt_kwargs)

        with tf.variable_scope('grad'):
            self._train_step, self.gvs = self._make_train_step(make_opt)

        # Metrics
        gradient_summaries(self.gvs)
        if nums is not None:
            self.gt_num_steps = tf.reduce_sum(nums, -1)
            num_step_per_sample = self.resample(self.num_step_per_sample)
            self.num_step_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.gt_num_steps, num_step_per_sample)))

        return self._train_step, tf.train.get_or_create_global_step()

    def _make_train_step(self, make_opt):

        self.baseline = self._make_baseline(self.cumulative_elbo_per_sample)

        num_steps_posterior = NumStepsDistribution(self.presence_prob[..., 0])
        posterior_num_steps_log_prob = num_steps_posterior.log_prob(self.num_step_per_sample)
        posterior_num_steps_log_prob = tf.reduce_sum(posterior_num_steps_log_prob, 0)

        if self.importance_resample:
            posterior_num_steps_log_prob = self.resample(posterior_num_steps_log_prob)
            posterior_num_steps_log_prob = tf.reshape(posterior_num_steps_log_prob, (self.batch_size, 1))

            baseline = tf.reshape(self.baseline, (self.effective_batch_size,))
            elbo_per_sample, baseline = self.resample(self.cumulative_elbo_per_sample, baseline)
            self.nelbo_per_sample = -tf.reshape(elbo_per_sample, (self.batch_size, 1))
            self.baseline = tf.reshape(baseline, (self.batch_size, 1))

            # this could be constant e.g. 1, but the expectation of this is zero anyway,
            #  so there's no point in adding that.
            r_imp_weight = 0.
        else:
            posterior_num_steps_log_prob = tf.reshape(posterior_num_steps_log_prob, (self.batch_size, self.iw_samples))
            r_imp_weight = self.cumulative_imp_weights
            self.nelbo_per_sample = -tf.reshape(self.cumulative_iw_elbo_per_sample, (self.batch_size, 1))

        num_steps_learning_signal = self.nelbo_per_sample
        self.nelbo = tf.reduce_mean(self.nelbo_per_sample)

        self.reinforce_loss = self._reinforce(num_steps_learning_signal - r_imp_weight, posterior_num_steps_log_prob)
        self.proxy_loss = (self.nelbo + self.reinforce_loss) / self.n_timesteps

        opt = make_opt(self.learning_rate)
        gvs = opt.compute_gradients(self.proxy_loss, var_list=self.model_vars)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        global_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(update_ops):
            train_step = opt.apply_gradients(gvs, global_step=global_step)

        return train_step, gvs

    def _make_baseline(self, per_sample_elbo):
        #####################

        if self.iw_samples == 1:
            return tf.zeros((self.batch_size, self.iw_samples), dtype=tf.float32)

        # compute the baseline
        #########################
        # 3) precompute the sum of per-sample bounds
        reshaped_per_sample_elbo = tf.reshape(per_sample_elbo, (self.batch_size, self.iw_samples))
        summed_per_sample_elbo = tf.reduce_sum(reshaped_per_sample_elbo, -1, keep_dims=True)

        # 4) compute the baseline
        all_but_one_average = (summed_per_sample_elbo - reshaped_per_sample_elbo) / (self.iw_samples - 1.)

        baseline, control = VIMCOEstimator._exped_baseline_and_control(reshaped_per_sample_elbo, all_but_one_average)
        baseline = tf.log(baseline) - tf.log(float(self.iw_samples)) + control
        return -baseline

    def _reinforce(self, learning_signal, posterior_num_steps_log_prob):
        """Implements REINFORCE for training the discrete probability distribution over number of steps and train-step
         for the baseline"""

        self.num_steps_learning_signal = learning_signal
        if self.baseline is not None:
            self.num_steps_learning_signal -= self.baseline

        axes = range(len(self.num_steps_learning_signal.get_shape()))
        imp_weight_mean, imp_weight_var = tf.nn.moments(self.num_steps_learning_signal, axes)
        tf.summary.scalar('imp_weight_mean', imp_weight_mean)
        tf.summary.scalar('imp_weight_var', imp_weight_var)
        reinforce_loss_per_sample = tf.stop_gradient(self.num_steps_learning_signal) * posterior_num_steps_log_prob

        shape = reinforce_loss_per_sample.shape.as_list()
        assert len(shape) == 2 and shape[0] == self.batch_size and shape[1] in (
        1, self.iw_samples), 'shape is {}'.format(shape)

        reinforce_loss = tf.reduce_mean(tf.reduce_sum(reinforce_loss_per_sample, -1))
        tf.summary.scalar('reinforce_loss', reinforce_loss)
        return reinforce_loss

    def resample(self, *args, **kwargs):
        axis = -1
        if 'axis' in kwargs:
            axis = kwargs['axis']
            del kwargs['axis']

        res = list(args)

        if self.iw_samples > 1:
            for i, arg in enumerate(res):
                res[i] = self._resample(arg, axis)

        if len(res) == 1:
            res = res[0]

        return res

    def _resample(self, arg, axis=-1):
        iw_sample_idx = self.imp_resampling_idx + tf.range(self.batch_size) * self.iw_samples
        return gather_axis(arg, iw_sample_idx, axis)

    def _log_resampled(self, resampled, name):
        resampled = self._resample(resampled)
        setattr(self, 'resampled_' + name, resampled)
        value = tf.reduce_mean(resampled)
        setattr(self, name, value)
        tf.summary.scalar(name, value)


# TODO: move somewhere else
def estimate_importance_weighted_elbo(batch_size, iw_samples, per_sample_elbo):
    per_sample_elbo = tf.reshape(per_sample_elbo, (batch_size, iw_samples))
    importance_weights = tf.nn.softmax(per_sample_elbo, -1)

    # tf.exp(tf.float32(89)) is inf, but if arg is 88 then it's not inf;
    # similarly on the negative, exp of -90 is 0;
    # when we subtract the max value, the dynamic range is about [-85, 0].
    # If we subtract 78 from control, it becomes [-85, 78], which is almost twice as big.
    control = tf.reduce_max(per_sample_elbo, -1, keep_dims=True) - 78.
    normalised = tf.exp(per_sample_elbo - control)
    iw_elbo = tf.log(tf.reduce_sum(normalised, -1, keep_dims=True)) + control - tf.log(float(iw_samples))
    iw_elbo = tf.squeeze(iw_elbo)
    return iw_elbo, importance_weights