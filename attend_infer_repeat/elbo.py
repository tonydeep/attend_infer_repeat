import functools
import tensorflow as tf
from tensorflow.contrib.distributions import Normal
from tensorflow.contrib.distributions.python.ops.kullback_leibler import kl as _kl

from prior import geometric_prior, tabular_kl


def kl_by_sampling(q, p, samples=None):

    if samples is None:
        samples = q.sample()
    return q.log_prob(samples) - tf.cast(p.log_prob(tf.cast(samples, p.dtype)), samples.dtype)


class AIRPriorMixin(object):

    scale_prior_loc = .5
    what_prior_scale = 1.
    where_prior_scale = 1.

    def _geom_success_prob(self, **kwargs):
        return 1e-5

    def _make_priors(self, **kwargs):
        """Defines prior distributions

        :return: prior over num steps, scale, shift and what
        """

        num_step_prior_prob, num_step_prior = geometric_prior(self._geom_success_prob(**kwargs), self.max_steps)
        scale = Normal(self.scale_prior_loc, self.where_prior_scale)
        shift = Normal(0., self.where_prior_scale)
        what = Normal(0., self.what_prior_scale)
        return num_step_prior_prob, num_step_prior, scale, shift, what


class KLBaseMixin(object):
    analytic_kl_expectation = False

    def _ordered_step_prob(self):
        raise NotImplementedError

    def _kl_where(self):
        raise NotImplementedError

    def _kl_what(self):
        raise NotImplementedError

    def _kl_num_steps(self):
        raise NotImplementedError


class KLZMixin(KLBaseMixin):

    def _ordered_step_prob(self):
        if self.analytic_kl_expectation:
            # reverse cumsum of q(n) needed to compute \E_{q(n)} [ KL[ q(z|n) || p(z|n) ]]
            ordered_step_prob = self.num_steps_posterior.prob()[..., 1:]
            ordered_step_prob = tf.cumsum(ordered_step_prob, axis=-1, reverse=True)
        else:
            ordered_step_prob = tf.squeeze(self.presence)
        return ordered_step_prob

    def _kl_what(self):
        what_kl = _kl(self.what_posterior, self.what_prior)
        what_kl = tf.reduce_sum(what_kl, -1) * self.ordered_step_prob
        what_kl_per_sample = tf.reduce_sum(what_kl, -1)
        return what_kl_per_sample

    def _kl_where(self):
        scale_kl = _kl(self.scale_posterior, self.scale_prior)
        shift_kl = _kl(self.shift_posterior, self.shift_prior)
        where_kl = tf.reduce_sum(scale_kl + shift_kl, -1) * self.ordered_step_prob
        where_kl_per_sample = tf.reduce_sum(where_kl, -1)
        return where_kl_per_sample, None, None


class KLNumStepsMixin(KLBaseMixin):
    def _kl_num_steps(self):
        num_steps_posterior_prob = self.num_steps_posterior.prob()
        steps_kl = tabular_kl(num_steps_posterior_prob, self.num_step_prior_prob)
        kl_num_steps_per_sample = tf.squeeze(tf.reduce_sum(steps_kl, 1))
        return kl_num_steps_per_sample


class KLMixin(KLZMixin, KLNumStepsMixin):
    pass


class KLBySamplingMixin(KLBaseMixin):
    def _ordered_step_prob(self):
        return tf.squeeze(self.presence)

    def _kl_what(self):
        what_kl = kl_by_sampling(self.what_posterior, self.what_prior, self.what)
        what_kl = tf.reduce_sum(what_kl, -1) * self.ordered_step_prob
        what_kl_per_sample = tf.reduce_sum(what_kl, -1)
        return what_kl_per_sample

    def _kl_where(self):
        ax = self.where.shape.ndims - 1
        scale, shift = tf.split(self.where, 2, ax)
        scale_kl = kl_by_sampling(self.scale_posterior, self.scale_prior, scale)
        shift_kl = kl_by_sampling(self.shift_posterior, self.shift_prior, shift)

        scale_kl, shift_kl = [tf.reduce_sum(i * self.ordered_step_prob[..., tf.newaxis], (-2, -1)) for i in (scale_kl, shift_kl)]
        where_kl_per_sample = scale_kl + shift_kl
        return where_kl_per_sample, scale_kl, shift_kl

    def _kl_num_steps(self):
        kl_num_steps_per_sample = kl_by_sampling(self.num_steps_posterior, self.num_step_prior, self.num_step_per_sample)
        return kl_num_steps_per_sample


class LogLikelihoodMixin(object):

    def _log_likelihood(self):
        # Reconstruction Loss, - \E_q [ p(x | z, n) ]
        rec_loss_per_sample = -self.output_distrib.log_prob(self.flat_used_obs)
        rec_loss_per_sample = tf.reduce_sum(rec_loss_per_sample, axis=(1, 2))
        return rec_loss_per_sample
