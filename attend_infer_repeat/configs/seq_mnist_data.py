import tensorflow as tf
from attrdict import AttrDict

from attend_infer_repeat.data import load_data as _load_data, tensors_from_data as _tensors
from attend_infer_repeat import tf_flags as flags

flags.DEFINE_string('train_path', 'seq_mnist_train.pickle', '')
flags.DEFINE_string('valid_path', 'seq_mnist_validation.pickle', '')
flags.DEFINE_integer('seq_len', 0, '')

axes = {'imgs': 1, 'labels': 0, 'nums': 1, 'coords': 1}


def truncate(data_dict, n_timesteps):
    data_dict['imgs'] = data_dict['imgs'][:n_timesteps]
    data_dict['coords'] = data_dict['coords'][:n_timesteps]
    return data_dict


def load(batch_size, n_timesteps=None):

    f = flags.FLAGS

    valid_data = _load_data(f.valid_path)
    train_data = _load_data(f.train_path)

    if n_timesteps is None and f.seq_len != 0:
        n_timesteps = f.seq_len

    if n_timesteps is not None:
        valid_data, train_data = [truncate(i, n_timesteps) for i in (valid_data, train_data)]

    train_tensors = _tensors(train_data, batch_size, axes, shuffle=True)
    valid_tensors = _tensors(valid_data, batch_size, axes, shuffle=False)

    n_timesteps = tf.shape(train_tensors['imgs'])[0]
    train_tensors['nums'] = tf.tile(train_tensors['nums'], (n_timesteps, 1, 1))
    valid_tensors['nums'] = tf.tile(valid_tensors['nums'], (n_timesteps, 1, 1))

    data_dict = AttrDict(
        train_img=train_tensors['imgs'],
        valid_img=valid_tensors['imgs'],
        train_num=train_tensors['nums'],
        valid_num=valid_tensors['nums'],
        train_tensors=train_tensors,
        valid_tensors=valid_tensors,
        train_data=train_data,
        valid_data=valid_data,
        axes=axes
    )

    return data_dict
