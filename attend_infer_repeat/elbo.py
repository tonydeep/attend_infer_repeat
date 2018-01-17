import tensorflow as tf


def kl_by_sampling(q, p, samples=None, q_weight=1., p_weight=1.):

    if samples is None:
        samples = q.sample()
    qp = q.log_prob(samples)
    pp = tf.cast(p.log_prob(tf.cast(samples, p.dtype)), samples.dtype)
    kl = q_weight * qp - p_weight * pp
    return kl


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