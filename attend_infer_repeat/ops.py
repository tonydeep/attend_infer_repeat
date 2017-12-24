import numpy as np
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
    # flat = tf.gather(tf.Print(flat, [flat], 'flat', -1, 100), tf.Print(linear_idx, [linear_idx], 'linear_idx', -1, 100))
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


def broadcast_against(tensor, against_expr):
    """Adds trailing dimensions to mask to enable broadcasting against data

    :param tensor: tensor to be broadcasted
    :param against_expr: tensor will be broadcasted against it
    :return: mask expr with tf.rank(mask) == tf.rank(data)
    """

    def cond(data, tensor):
        return tf.less(tf.rank(tensor), tf.rank(data))

    def body(data, tensor):
        return data, tf.expand_dims(tensor, -1)

    shape_invariants = [against_expr.get_shape(), tf.TensorShape(None)]
    _, tensor = tf.while_loop(cond, body, [against_expr, tensor], shape_invariants)
    return tensor


# def select_present(x, presence, batch_size=1, name='select_present'):
#     with tf.variable_scope(name):
#         presence = 1 - tf.to_int32(presence)  # invert mask
#
#         # bs = x.get_shape()[0]
#         # if bs != None:  # here type(bs) is tf.Dimension and == is ok
#         #     batch_size = int(bs)
#
#         x_shape = tf.shape(x)
#         p_shape = tf.shape(presence)
#         sample_shape = x_shape[tf.shape(p_shape)[0]:]
#         # batch_size = tf.reduce_prod(tf.shape(presence))
#         # batch_size = 3
#
#         presence = tf.reshape(presence, [-1])
#         flat_shape = tf.concat(([-1], sample_shape), 0)
#         x = tf.reshape(x, flat_shape)
#
#         # 2 partitions for every sample in a batch: 0 and 1
#         num_partitions = 2 * batch_size
#         r = tf.range(0, num_partitions,  2)
#         r.set_shape(tf.TensorShape(batch_size))
#         r = broadcast_against(r, presence)
#         # r = _broadcast_against(presence, r)
#
#         presence += r
#
#         selected = tf.dynamic_partition(x, presence, num_partitions)
#         # selected = tf.concat(selected, axis=0)
#         # selected = tf.reshape(selected, x_shape)
#
#     return presence
# [ 0  2  4  6  8 10]
# [ 1  2  4  7  9 10]
# [0, 1, 1, 0, 0, 1]

def select_present(x, presence, batch_size=None, name='select_present'):
    with tf.variable_scope(name):
        presence = 1 - tf.to_int32(presence)  # invert mask

        if batch_size is None:
            batch_size = int(x.shape[0])

        num_partitions = 2 * batch_size
        r = tf.range(0, num_partitions,  2)
        r.set_shape(tf.TensorShape(batch_size))
        r = broadcast_against(r, presence)
        presence += r

        selected = tf.dynamic_partition(x, presence, num_partitions)
        selected = tf.concat(selected, 0)
        selected = tf.reshape(selected, tf.shape(x))

    return selected


def select_present_list(tensor_list, presence, batch_size=None, name='select_present_many'):
    """Like `select_present`, but handles a list of tensors.

     It concatenates the tensors along the last dimension, calls `select_present` only once
     and splits the tensors again. It's faster and the graph is less complicated that
     way.

    :param tensor_list:
    :param presence:
    :param batch_size:
    :param name:
    :return:
    """
    orig_inpt = tensor_list
    with tf.variable_scope(name):
        tensor_list = nest.flatten(tensor_list)
        lens = [0] + [int(t.shape[-1]) for t in tensor_list]
        lens = np.cumsum(lens)

        merged = tf.concat(tensor_list, -1)
        merged = select_present(merged, presence, batch_size)
        tensor_list = []

        for i in xrange(len(lens) - 1):
            st, ed = lens[i], lens[i + 1]
            tensor_list.append(merged[..., st:ed])

    return nest.pack_sequence_as(structure=orig_inpt, flat_sequence=tensor_list)


        # def scatter_present(n, vals, elem_shape):
#     # i = tf.range(tf.to_int32(n))[tf.newaxis]
#     vals = vals[tf.newaxis]
#
#     # scattered = tf.scatter_nd(i, vals, elem_shape)
#
#     rank = tf.shape(elem_shape)[0]
#     before = tf.zeros((rank, 1), dtype=tf.int32)
#     after = tf.to_int32(elem_shape) - tf.to_int32(tf.shape(vals))
#     padding = tf.concat([before, after[:, tf.newaxis]], 1)
#     scattered = tf.pad(vals, padding)
#
#     scattered = tf.cond(tf.greater(n, 0.), lambda: scattered, lambda: tf.zeros(elem_shape, tf.float32))
#
#     # return tf.to_float(tf.rank(vals))
#     # return tf.to_float(padding)
#     return scattered


# def select_present2(x, presence, batch_size=1, name='select_present'):
    # with tf.variable_scope(name):
    #
    #     bool_pres = tf.cast(presence, bool)
    #     idx = tf.where(bool_pres)
    #     idx = tf.cast(idx, tf.int32)
    #     # values = tf.gather_nd(x, idx)
    #     values = tf.gather(x, idx)
    #
    #     nnz = tf.reduce_sum(presence, -1)
    #     elem_shape = tf.shape(x)[1:]
    #
    #     def map_fn((n, vals)):
    #         return scatter_present(n, vals, elem_shape)
    #
    #     # idx2 = tf.to_int32(tf.where(tf.greater(nnz, 0.)))
    #     # values = tf.scatter_nd(idx2, values, tf.shape(nnz))
    #
    #     return nnz, elem_shape, values, idx
    #
    #     selected = tf.map_fn(map_fn, [nnz, values], dtype=tf.float32)
    #     # selected = tf.reshape(selected, tf.shape(x))
    #
    #     # return values, elem_shape, selected
    #     # selected.set_shape(x.get_shape())
    #     #
    #     #
    #
    #     # new_idx = tf.range(tf.to_int32(nnz))
    #     #
    #     # selected = tf.scatter_nd(new_idx, values, tf.shape(x))
    #     # # return nnz, idx, values, selected
    #     return selected


def select_present_impl((vals, p)):
    idx = tf.where(tf.cast(p, bool))
    selected_vals = tf.gather_nd(vals, idx)

    # selected_vals = gather_axis(vals, idx, 0)
    # selected_vals = tf.boolean_mask(vals, tf.cast(p, bool))

    nnz = tf.to_int32(tf.reduce_sum(p))
    scatter_idx = tf.range(nnz)
    scattered = tf.scatter_nd(scatter_idx, selected_vals, tf.shape(vals))
    return scattered, scatter_idx, selected_vals, tf.shape(vals), tf.shape(idx), tf.shape(vals)
    # return selected_vals, scatter_idx, selected_vals, tf.shape(vals), tf.shape(idx), tf.shape(vals)


def select_present2(x, presence, batch_size=1, name='select_present'):

    batch_size = 64
    try:
        batch_size = int(x.shape[0])
    except TypeError:
        pass

    selected = tf.map_fn(select_present_impl, [x, presence], dtype=tf.float32, parallel_iterations=batch_size)
    return selected


def compute_object_ids(last_used_id, prev_ids, propagated_pres, discovery_pres):
    last_used_id, prev_ids, propagated_pres, discovery_pres = [tf.convert_to_tensor(i) for i in (last_used_id, prev_ids, propagated_pres, discovery_pres)]
    prop_ids = prev_ids * propagated_pres - (1 - propagated_pres)

    # each object gets a new id
    id_increments = tf.cumsum(discovery_pres, 1)
    # find the new id by incrementing the last used ids
    disc_ids = id_increments + last_used_id[:, tf.newaxis]

    # last used ids needs to be incremented by the maximum value
    last_used_id += id_increments[:, -1]

    disc_ids = disc_ids * discovery_pres - (1 - discovery_pres)
    new_ids = tf.concat([prop_ids, disc_ids], 1)
    return last_used_id, new_ids


def update_num_obj_counts(num_counts, obj_counts):
    batch_size = int(obj_counts.shape[0])
    obj_counts = tf.expand_dims(tf.to_int32(obj_counts), -1)
    batch_idx = tf.expand_dims(tf.range(batch_size), -1)
    obj_count_idx = tf.concat((batch_idx, obj_counts), -1)
    count_updates = tf.scatter_nd(obj_count_idx, tf.ones([batch_size]), tf.shape(num_counts))
    return num_counts + count_updates