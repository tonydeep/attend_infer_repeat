import math
import tensorflow as tf


def vimco_baseline(per_sample_elbo):

    n_iw_samples = int(per_sample_elbo.shape[-1])
    summed_per_sample_elbo = tf.reduce_sum(per_sample_elbo, -1, keep_dims=True)
    all_but_one_average = (summed_per_sample_elbo - per_sample_elbo) / (n_iw_samples - 1.)

    diag = tf.matrix_diag(all_but_one_average - per_sample_elbo)
    baseline = per_sample_elbo[..., tf.newaxis] + diag
    return tf.reduce_logsumexp(baseline, -2) - math.log(float(n_iw_samples))
