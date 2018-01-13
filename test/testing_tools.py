import os
import unittest
import tensorflow as tf
from tensorflow.python.util import nest


class TFTestBase(unittest.TestCase):

    vars = {
        'x': [tf.float32, None],
        'y': [tf.float32, None],
        'm': [tf.float32, None]
    }

    @classmethod
    def setUpClass(cls):
        tf.reset_default_graph()

        for k, v in cls.vars.iteritems():
            setattr(cls, k, tf.placeholder(v[0], v[1], name=k))
        cls.sess = tf.Session()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.sess.close()
        except AttributeError:
            pass
        tf.reset_default_graph()

    @classmethod
    def init_vars(cls):
        cls.sess.run(tf.global_variables_initializer())
        cls.sess.run(tf.local_variables_initializer())

    def feed_dict(self, xx=None, yy=None, mm=None):
        fd = {}

        if xx is not None:
            fd[self.x] = xx

        if yy is not None:
            fd[self.y] = yy

        if mm is not None:
            fd[self.m] = mm

        return fd

    def eval(self, expr, xx=None, yy=None, mm=None, feed_dict=None):
        if feed_dict is None:
            feed_dict = self.feed_dict(xx, yy, mm)

        try:
            expr = tf.convert_to_tensor(expr)
        except TypeError: pass

        return self.sess.run(expr, feed_dict)


def test_path(path=None):
    p = os.path.abspath(os.path.dirname(__file__))
    if path is not None:
        p = os.path.join(p, path)
    return p


def flatten_check(x, check_func):
    res = []
    x = nest.flatten(x)
    for y in x:
        res.append(check_func(y).any())
    return any(res)


def trainable_model_vars(skip_vars=tuple()):
    vars = tf.trainable_variables()
    model_vars = sorted(list(set(vars) - set(skip_vars)), key=lambda v: v.name)
    return model_vars


def print_trainable_variables(model_name, skip_vars=tuple()):
    model_vars = trainable_model_vars(skip_vars)
    print '{} Trainable Variables for model {}'.format(len(model_vars), model_name)
    for v in model_vars:
        print v.name, v.shape.as_list()
    return model_vars