import time
import unittest

from attrdict import AttrDict

import numpy as np
from numpy.testing import assert_array_less, assert_array_equal, assert_array_almost_equal

import tensorflow as tf
import sonnet as snt

from attend_infer_repeat.mnist_model import MNISTPriorMixin
from attend_infer_repeat.modules import Decoder, Encoder, StepsPredictor, StochasticTransformParam
from attend_infer_repeat.model import APDRModel
from testing_tools import print_trainable_variables


class APDRModelMock(APDRModel, MNISTPriorMixin):
    importance_resample = True
    transition_class = snt.VanillaRNN
    time_transition_class = snt.GRU
    prior_rnn_class = snt.LSTM
    decode_prop = True
    constant_prop_prior = 10.
    propagate_disc_what = True
    # anneal_iter = 2000
    # anneal_temp = 2


def make_modules():

    return dict(
        transition=snt.VanillaRNN(3),
        input_encoder=(lambda: Encoder(5)),
        glimpse_encoder=(lambda: Encoder(7)),
        glimpse_decoder=(lambda x: Decoder(11, x)),
        transform_estimator=(lambda: StochasticTransformParam(13, scale_bias=-3.)),
        steps_predictor=(lambda: StepsPredictor(17, steps_bias=2.))
    )


class APDRModelTest(unittest.TestCase):
    learning_rate = 1e-4
    batch_size = 10
    img_size = (5, 7)
    crop_size = (2, 2)
    n_what = 13
    n_steps_per_image = 3
    iw_samples = 2
    n_timesteps = 3

    @classmethod
    def setUpClass(cls):
        tf.reset_default_graph()

        cls.timer_dict = dict()

        cls.imgs = tf.placeholder(tf.float32, (cls.n_timesteps, cls.batch_size,) + cls.img_size, name='inpt')
        cls.nums = tf.placeholder(tf.float32, (cls.n_timesteps, cls.batch_size, cls.n_steps_per_image), name='nums')

        print 'Building AIR'
        cls.modules = make_modules()
        time_start = time.clock()
        cls.air = APDRModelMock(cls.imgs, cls.n_steps_per_image, cls.crop_size, cls.n_what,
                                iw_samples=cls.iw_samples,
                                **cls.modules
                                )

        cls.rnn_outputs = AttrDict({k: getattr(cls.air, k) for k in cls.air.rnn_output_names})
        cls.outputs = AttrDict({k: getattr(cls.air, k) for k in cls.air.output_names})
        print 'Constructed model'

        cls.train_step = cls.air.train_step(cls.learning_rate, nums=cls.nums)
        cls.loss = tf.reduce_mean(cls.air.iw_elbo)
        cls.loss = cls.air.nelbo / cls.air.n_timesteps

        cls.kl = tf.reduce_mean(cls.air.kl_div)
        cls.likelihood = tf.reduce_mean(cls.air.rec_loss)

        print 'Computed gradients'
        cls.register_time(time_start, time.clock(), 'Building model')

        cls.model_vars = print_trainable_variables(cls.__name__)

    @classmethod
    def tearDownClass(cls):
        for k, v in cls.timer_dict.iteritems():
            print '{} took {}s'.format(k, v)
        super(APDRModelTest, cls).tearDownClass()

    @classmethod
    def register_time(cls, start_time, end_time, name):
        cls.timer_dict[name] = end_time - start_time

    def test_forward(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        xx = np.random.rand(*self.imgs.get_shape().as_list())

        time_start = time.clock()

        rnn_outputs, outputs, l = sess.run([self.rnn_outputs, self.outputs, [self.loss, self.kl, self.likelihood]], {self.imgs: xx})

        self.register_time(time_start, time.clock(), 'Forward pass')


        # Prop probability should be always zero at the first timestep
        assert_array_equal(outputs.prop_prob[0], 0.)

        # Number of steps; should be less or equal than the max specified
        num_step = outputs.num_step_per_sample
        num_disc_step = outputs.num_disc_step_per_sample
        num_prop_step = outputs.num_prop_step_per_sample
        summed_num_step = num_disc_step + num_prop_step

        assert_array_less(num_step, self.n_steps_per_image + 1)
        assert_array_less(num_disc_step, num_step + 1)
        assert_array_less(num_prop_step, num_step + 1)

        ## values should be discrete
        assert_array_equal(num_step, np.round(num_step))
        assert_array_equal(num_disc_step, np.round(num_disc_step))
        assert_array_equal(num_prop_step, np.round(num_prop_step))

        ## prop + disc should be equal to overall value;
        assert_array_almost_equal(summed_num_step, num_step)

        # Test KL
        ## Prop step KL should be zero at the first timestep
        assert_array_equal(outputs.kl_prop_steps_per_sample[0], 0.)

        ## individual values can be negative due to sampling, but the mean should be positive
        kls = 'where what steps prop_steps disc_steps'.split()
        for kl_name in kls:
            kl_key = 'kl_{}_per_sample'.format(kl_name)
            kl_value = outputs[kl_key]
            self.assertGreaterEqual(kl_value.mean(), 0., 'KL_{}: min = {}, mean = {}'.format(kl_name, kl_value.min(), kl_value.mean()))

        assert_array_equal(outputs.kl_prop_steps_per_sample + outputs.kl_disc_steps_per_sample, outputs.kl_steps_per_sample)

        print 'rnn_outputs:'
        for k, v in rnn_outputs.iteritems():
            self.assertEqual(v.shape[:2], (self.n_timesteps, self.iw_samples * self.batch_size),
                             'Invalid shape of {} in rnn_output "{}"'.format(v.shape, k))
            print k, v.shape

        print
        print 'outputs:'
        for k, v in outputs.iteritems():
            self.assertEqual(v.shape[0], self.n_timesteps, 'Invalid shape of {} in output "{}"'.format(v.shape, k))
            print k, v.shape

        # check that there are ids different from -1 and that at least some ids are propagated
        # print 'obj_ids'
        fraction_propagated = 0.
        fraction_no_objects = 0.
        for i in xrange(self.batch_size):
            unique_ids = set()
            prop_ids = set()

            for t in xrange(self.n_timesteps):
                ids = set(outputs.obj_id[t, i].squeeze())
                prop_ids = prop_ids.union(ids.intersection(unique_ids))
                disc_ids = ids - unique_ids
                unique_ids = unique_ids.union(ids)

                # print 'i={}, t={}, id={}, p={} prop_pres={}, disc_pres={}'\
                #     .format(i, t, outputs.obj_id[t, i].squeeze(), rnn_outputs.presence[t, i].squeeze(),
                #             outputs.prop_pres[t, i].squeeze(), outputs.disc_pres[t, i].squeeze())

                print 't', t, 'num_disc', len(disc_ids), disc_ids

            self.assertGreater(len(unique_ids), 1.)  # at least one object was discovered
            if -1 in unique_ids:
                fraction_no_objects += 1

            prop_ids = prop_ids - {-1.}
            if len(prop_ids) > 0.:
                fraction_propagated += 1.

        self.assertTrue(float(fraction_no_objects) / self.batch_size, .3)  # sometimes not all objects are present
        self.assertGreater(float(fraction_propagated) / self.batch_size, .3) # check that sometimes SOME object are propagated

        print 'what_loc', rnn_outputs.what_loc.mean()
        print 'what_scale', rnn_outputs.what_scale.mean()
        print 'where_loc', rnn_outputs.where_loc.mean()
        print 'where_scale', rnn_outputs.where_scale.mean()

        # print
        print 'loss = {}, kl = {}, likelihood = {}'.format(*l)
        self.assertLess(l[0], 100.)
        self.assertGreater(l[0], 20.)

    def test_backward(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        xx = np.random.rand(*self.imgs.get_shape().as_list())

        print 'Running train step'

        time_start = time.clock()

        sess.run(self.train_step, {self.imgs: xx})
        self.register_time(time_start, time.clock(), 'Backward pass')
        print 'Done'

    def test_learning_signal_shape(self):
        learning_signal_shape = self.air.num_steps_learning_signal.shape.as_list()
        self.assertEqual(learning_signal_shape, [self.batch_size, 1])

    def test_num_vars(self):
        """This is to ensure test failure whenever number of variables increases: it forces me to think why it happened
        as I need to change this number explicitly to make tests run agian."""
        self.assertEqual(len(self.model_vars), 63)



# # Naive implt
# Building model took 27.32111s 26.776982s 27.426379s
# Backward pass took 1.70163s 1.665743s 1.711467s
# Forward pass took 0.33428s 0.338653s 0.348015s
#
# # Mering impl:
# Building model took 22.111965s 21.519563s 21.664563s
# Backward pass took 1.366654s 1.505577s 1.358077s
# Forward pass took 0.293968s 0.28257s 0.309403s
#
# # No partitioning:
# Building model took 21.181945s 21.53531s 21.635552s
# Backward pass took 1.303993s 1.281379s 1.293466s
# Forward pass took 0.303936s 0.285393s 0.292647s
#
# ## Conclusions:
# Partitioning is an expensive operation and we should avoid calling it whenever we can
# but we use it if we have to, and for now there's no other alternative
