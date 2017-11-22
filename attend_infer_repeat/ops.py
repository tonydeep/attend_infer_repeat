import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.util import nest


class Loss(object):
    """Helper class for keeping track of losses"""

    def __init__(self):
        self._value = None
        self._per_sample = None

    def add(self, loss=None, per_sample=None, weight=1.):
        if isinstance(loss, Loss):
            per_sample = loss.per_sample
            loss = loss.value

        self._update('_value', loss, weight)
        self._update('_per_sample', per_sample, weight)

    def _update(self, name, expr, weight):
        value = getattr(self, name)
        expr *= weight
        if value is None:
            value = expr
        else:
            assert value.get_shape().as_list() == expr.get_shape().as_list(), 'Shape should be {} but is {}'.format(value.get_shape(), expr.get_shape())
            value += expr

        setattr(self, name, value)

    def _get_value(self, name):
        v = getattr(self, name)
        if v is None:
            v = tf.zeros([])
        return v

    @property
    def value(self):
        return self._get_value('_value')

    @property
    def per_sample(self):
        return self._get_value('_per_sample')


def make_moving_average(name, value, init, decay, log=True):
    """Creates an exp-moving average of `value` and an update op, which is added to UPDATE_OPS collection.

    :param name: string, name of the created moving average tf.Variable
    :param value: tf.Tensor, the value to be averaged
    :param init: float, an initial value for the moving average
    :param decay: float between 0 and 1, exponential decay of the moving average
    :param log: bool, add a summary op if True
    :return: tf.Tensor, the moving average
    """
    var = tf.get_variable(name, shape=value.get_shape(),
                          initializer=tf.constant_initializer(init), trainable=False)

    update = moving_averages.assign_moving_average(var, value, decay, zero_debias=False)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update)
    if log:
        tf.summary.scalar(name, var)

    return var


def clip_preserve(expr, min, max):
    """Clips the immediate gradient but preserves the chain rule

    :param expr: tf.Tensor, expr to be clipped
    :param min: float
    :param max: float
    :return: tf.Tensor, clipped expr
    """
    clipped = tf.clip_by_value(expr, min, max)
    return tf.stop_gradient(clipped - expr) + expr


def anneal_weight(init_val, final_val, anneal_type, global_step, anneal_steps, hold_for=0., steps_div=1.,
                   dtype=tf.float64):
    val, final, step, hold_for, anneal_steps, steps_div = (tf.cast(i, dtype) for i in
                                                           (init_val, final_val, global_step, hold_for, anneal_steps,
                                                            steps_div))
    step = tf.maximum(step - hold_for, 0.)

    if anneal_type == 'exp':
        decay_rate = tf.pow(final / val, steps_div / anneal_steps)
        val = tf.train.exponential_decay(val, step, steps_div, decay_rate)

    elif anneal_type == 'linear':
        val = final + (val - final) * (1. - step / anneal_steps)
    else:
        raise NotImplementedError

    anneal_weight = tf.maximum(final, val)
    return anneal_weight


def sample_from_1d_tensor(arr, idx):
    """Takes samples from `arr` indicated by `idx`

    :param arr:
    :param idx:
    :return:
    """
    arr = tf.convert_to_tensor(arr)
    assert len(arr.get_shape()) == 1, "shape is {}".format(arr.get_shape())

    idx = tf.to_int32(idx)
    arr = tf.gather(tf.squeeze(arr), idx)
    return arr


def sample_from_tensor(tensor, idx):
    """Takes sample from `tensor` indicated by `idx`, works for minibatches

    :param tensor:
    :param idx:
    :return:
    """
    tensor = tf.convert_to_tensor(tensor)

    assert tensor.shape.ndims == (idx.shape.ndims + 1) \
           or ((tensor.shape.ndims == idx.shape.ndims) and (idx.shape[-1] == 1)), \
        'Shapes: tensor={} vs idx={}'.format(tensor.shape.ndims, idx.shape.ndims)

    batch_shape = tf.shape(tensor)[:-1]
    trailing_dim = int(tensor.shape[-1])
    n_elements = tf.reduce_prod(batch_shape)
    shift = tf.range(n_elements) * trailing_dim

    tensor_flat = tf.reshape(tensor, (-1,))
    idx_flat = tf.reshape(tf.to_int32(idx), (-1,)) + shift
    samples_flat = sample_from_1d_tensor(tensor_flat, idx_flat)
    samples = tf.reshape(samples_flat, batch_shape)

    return samples


def expand_around_zero(x, eps):
    gaus = tf.exp(-0.5 * x ** 2)
    sign = tf.to_float(tf.greater_equal(x, 0.))
    return x + (2 * sign - 1.) * eps * gaus 


def gather_axis(tensor, idx, axis=-1):
    """Gathers indices `idx` from `tensor` along axis `axis`

    The shape of the returned tensor is as follows:
    >>> shape = tensor.shape
    >>> shape[axis] = len(idx)
    >>> return shape

    :param tensor: n-D tf.Tensor
    :param idx: 1-D tf.Tensor
    :param axis: int
    :return: tf.Tensor
    """

    axis = tf.convert_to_tensor(axis)
    neg_axis = tf.less(axis, 0)
    axis = tf.cond(neg_axis, lambda: tf.shape(tf.shape(tensor))[0] + axis, lambda: axis)
    shape = tf.shape(tensor)
    pre, post = shape[:axis+1], shape[axis+1:]
    shape = tf.concat((pre[:-1], tf.shape(idx)[:1], post), -1)

    n = tf.reduce_prod(pre[:-1])
    idx = tf.tile(idx[tf.newaxis], (n, 1))
    idx += tf.range(n)[:, tf.newaxis] * pre[-1]
    linear_idx = tf.reshape(idx, [-1])

    flat = tf.reshape(tensor, tf.concat(([n * pre[-1]], post), -1))
    flat = tf.gather(flat, linear_idx)
    tensor = tf.reshape(flat, shape)
    return tensor


def tile_input_for_iwae(tensor, iw_samples, with_time=False):
    """Tiles tensor `tensor` in such a way that tiled samples are contiguous in memory;
    i.e. it tiles along the axis after the batch axis and reshapes to have the same rank as
    the original tensor

    :param tensor: tf.Tensor to be tiled
    :param iw_samples: int, number of importance-weighted samples
    :param with_time: boolean, if true than an additional axis at the beginning is assumed
    :return:
    """

    shape = tensor.shape.as_list()
    if with_time:
        shape[0] = tf.shape(tensor)[0]
    shape[with_time] *= iw_samples

    tiles = [1, iw_samples] + [1] * (tensor.shape.ndims - (1 + with_time))
    if with_time:
        tiles = [1] + tiles

    tensor = tf.expand_dims(tensor, 1 + with_time)
    tensor = tf.tile(tensor, tiles)
    tensor = tf.reshape(tensor, shape)
    return tensor


def stack_states(states):
    orig_state = states[0]
    states = [nest.flatten(s) for s in states]
    states = zip(*states)
    for i, state in enumerate(states):
        states[i] = tf.stack(state, 1)
    states = nest.pack_sequence_as(orig_state, states)
    return states

def maybe_getattr(obj, name):
    attr = None
    if name is not None:
        attr = getattr(obj, name, None)
    return attr
