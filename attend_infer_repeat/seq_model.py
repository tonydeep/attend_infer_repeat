import functools

import tensorflow as tf

from elbo import estimate_importance_weighted_elbo
from evaluation import gradient_summaries
from grad import VIMCOEstimator
from modules import AIRDecoder
from ops import tile_input_for_iwae, gather_axis
from prior import NumStepsDistribution


# TODO: implement FIVO & per-timestep VIMCO for FIVO


class SeqAIRModel(object):
    """Generic AIR model

    :param analytic_kl_expectation: bool, computes expectation over conditional-KL analytically if True
    """

    time_transition_class = None

    def __init__(self, obs, max_steps, glimpse_size,
                 n_what, transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
                 steps_predictor,
                 output_std=1., output_multiplier=1., iw_samples=1,
                 condition_on_prev=False, prior_around_prev=False,
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

        self.condition_on_prev = condition_on_prev
        self.prior_around_prev = prior_around_prev

        self.debug = debug

        shape = self.obs.get_shape().as_list()
        self.n_timesteps = shape[0]
        self.batch_size = shape[1]

        self.img_size = shape[2:]
        self.effective_batch_size = self.batch_size * self.iw_samples
        self.used_obs = tile_input_for_iwae(obs, self.iw_samples, with_time=True)

        with tf.variable_scope(self.__class__.__name__):
            self.output_multiplier = tf.Variable(output_multiplier, dtype=tf.float32, trainable=False,
                                                 name='canvas_multiplier')

            # save existing variables to know later what we've created
            previous_vars = tf.trainable_variables()

            self._build(glimpse_decoder, transition, input_encoder, glimpse_encoder, transform_estimator,
                        steps_predictor, **cell_kwargs)

            # group variables
            model_vars = set(tf.trainable_variables()) - set(previous_vars)
            self.decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                  scope=self.decoder.variable_scope.name)
            self.encoder_vars = list(model_vars - set(self.decoder_vars))
            self.model_vars = list(model_vars)

    def _build(self, glimpse_decoder, *cell_args, **cell_kwargs):
        """Build the model. See __init__ for argument description"""

        self.num_step_prior_prob, self.num_step_prior, \
        self.scale_prior, self.shift_prior, self.what_prior = self._make_priors()

        self.decoder = AIRDecoder(self.img_size, self.glimpse_size, glimpse_decoder, batch_dims=2)

        self.cells = self._make_cells(*cell_args, **cell_kwargs)
        self.cell = self.cells[0]

        res = self._time_loop()
        self.final_state = res[2]
        self.cumulative_imp_weights = res[4]
        tas = res[7:]

        # TODO: prettify
        self.output_names = 'obj_id canvas glimpse posterior_step_prob likelihood_per_sample kl_what_per_sample' \
                            ' kl_where_per_sample kl_steps_per_sample kl_per_sample elbo_per_sample num_step_per_sample' \
                            ' importance_weight iw_elbo'.split()
        for name, ta in zip(self.cell.output_names + self.output_names, tas):
            output = ta.stack()
            setattr(self, name, output)

        self.cumulative_elbo_per_sample = tf.reduce_sum(self.elbo_per_sample, 0)

        self.cumulative_iw_elbo_per_sample, _ = estimate_importance_weighted_elbo(self.batch_size, self.iw_samples,
                                                                                  self.cumulative_elbo_per_sample)

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
        resampled_names = 'obj_id canvas glimpse presence where posterior_step_prob'.split()
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
        shape = arg.shape.as_list()
        shape[axis] = self.batch_size
        resampled = gather_axis(arg, iw_sample_idx, axis)
        resampled.set_shape(shape)
        return resampled

    def _log_resampled(self, resampled, name):
        resampled = self._resample(resampled)
        setattr(self, 'resampled_' + name, resampled)
        value = tf.reduce_mean(resampled)
        setattr(self, name, value)
        tf.summary.scalar(name, value)
