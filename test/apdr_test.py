import unittest

from numpy.testing import assert_array_equal, assert_array_less

from attend_infer_repeat.cell import AIRCell, PropagatingAIRCell
from attend_infer_repeat.modules import *
from attend_infer_repeat.refactor import APDR
from attend_infer_repeat.refactor import AttendDiscoverRepeat, AttendPropagateRepeat
from testing_tools import flatten_check, print_trainable_variables


def make_modules(step_bias=0.):
    return dict(
        transition=snt.VanillaRNN(3),
        input_encoder=(lambda: Encoder(5)),
        glimpse_encoder=(lambda: Encoder(7)),
        transform_estimator=(lambda: StochasticTransformParam(13)),
        steps_predictor=(lambda: StepsPredictor(17, step_bias))
    )


class ModuleTest(object):
    @classmethod
    def setUpClass(cls):
        super(ModuleTest, cls).setUpClass()
        tf.reset_default_graph()

        vars_before = tf.trainable_variables()

        cls.batch_size = 10
        cls.img_size = (5, 7)
        cls.crop_size = (2, 2)
        cls.n_latent = 13
        cls.n_steps = 3
        cls.step_success_prob = .9

        cls.img = tf.placeholder(tf.float32, (cls.batch_size,) + cls.img_size, name='inpt')

        cls.modules = make_modules()
        cls.air_cell, cls.model, cls.output = cls._make_model()

        cls.sess = tf.Session()
        cls.sess.run(tf.global_variables_initializer())

        print_trainable_variables(cls.__name__, vars_before)

    def test_fields(self):
        self.assertEqual(self.model._n_steps, self.n_steps)
        self.assertEqual(self.model._batch_size, self.batch_size)
        self.assertEqual(self.model._cell, self.air_cell)


class DiscoverTest(ModuleTest, unittest.TestCase):
    @classmethod
    def _make_model(cls):
        air_cell = AIRCell(cls.img_size, cls.crop_size, cls.n_latent, condition_on_inpt=False, **cls.modules)

        # cls.num_present_objects = tf.zeros((cls.batch_size,), name='num_present_objects')
        cls.num_present_objects = tf.to_float(tf.random_uniform((cls.batch_size,), 0, cls.n_steps + 1,
                                                                tf.int32, name='num_present_objects'))
        cls.conditioning = None

        model = AttendDiscoverRepeat(cls.n_steps, cls.batch_size, air_cell, cls.step_success_prob)
        output = model(cls.img, cls.num_present_objects, cls.conditioning)

        return air_cell, model, output

    def test_shapes(self):
        num_hidden_outputs = 9
        num_other_outputs = 8

        self.assertEqual(len(self.output.hidden_outputs), num_hidden_outputs)
        self.assertEqual(len(self.output), num_other_outputs + len(self.output.hidden_outputs))
        for k, v in self.output.iteritems():
            if k not in 'hidden_outputs num_steps_prob':
                shape = v.shape.as_list()
                self.assertEqual(shape[0], self.batch_size, '{} has incorrect shape={}'.format(k, v.shape))
                if len(shape) > 1:
                    self.assertEqual(shape[1], self.n_steps, '{} has incorrect shape={}'.format(k, v.shape))

        assert_array_equal(self.output.num_steps_prob.shape.as_list(), [self.batch_size, self.n_steps + 1])

    def test_values(self):

        random_img = np.random.uniform(size=self.img.shape.as_list())
        values, num_present_objects = self.sess.run([self.output, self.num_present_objects], {self.img: random_img})
        max_allowed_objects = self.n_steps - num_present_objects

        assert_array_less(values.num_steps, max_allowed_objects + 1)

        for k, v in values.iteritems():
            self.assertFalse(flatten_check(v, np.isnan), 'NaNs in {}'.format(k))
            self.assertFalse(flatten_check(v, np.isinf), 'infs in {}'.format(k))

        # KL is estimated by sampling, but mean should be nonnegative
        self.assertGreaterEqual(values.kl.mean(), 0.)
        self.assertGreaterEqual(values.kl_what.mean(), 0.)
        self.assertGreaterEqual(values.kl_where.mean(), 0.)
        self.assertGreaterEqual(values.kl_num_step.mean(), 0.)

        #       KL what & where should be equal to zero for objects that are not there
        presence = values.presence[..., 0]
        assert_array_equal(values.kl_what, values.kl_what * presence)
        assert_array_equal(values.kl_where, values.kl_where * presence)


class PropagateTest(ModuleTest, unittest.TestCase):
    @classmethod
    def _make_model(cls):
        air_cell = PropagatingAIRCell(cls.img_size, cls.crop_size, cls.n_latent, **cls.modules)
        cls.prior_cell = snt.GRU(3)

        cls.init_rnn_state = cls.prior_cell.initial_state(cls.batch_size, tf.float32, trainable=True)
        cls.prior_rnn_state = tf.tile(tf.expand_dims(cls.init_rnn_state, -2), (1, cls.n_steps, 1))

        cls.num_obj_counts = tf.ones((cls.batch_size, cls.n_steps + 1))
        cls.prior_state = [cls.prior_rnn_state, cls.num_obj_counts]

        cls.what_tm1 = tf.random_normal((cls.batch_size, cls.n_steps, cls.n_latent), name='what_tm1')
        cls.where_tm1 = tf.random_normal((cls.batch_size, cls.n_steps, 4), name='where_tm1')
        cls.pres_tm1 = tf.to_float(tf.random_uniform((cls.batch_size, cls.n_steps, 1), 0, 2, tf.int32, name='pres_tm1'))

        mean = 3 * cls.pres_tm1 - 3 * (1. - cls.pres_tm1)
        cls.pres_logit_tm1 = tf.random_normal((cls.batch_size, cls.n_steps, 1)) + mean
        cls.z_tm1 = [cls.what_tm1, cls.where_tm1, cls.pres_tm1, cls.pres_logit_tm1]

        cls.temporal_cell = snt.LSTM(5)
        cls.temporal_state = cls.temporal_cell.initial_state(cls.batch_size, tf.float32, trainable=True)
        cls.temporal_conditioning, _ = cls.temporal_cell(cls.temporal_state[0], cls.temporal_state)

        model = AttendPropagateRepeat(cls.n_steps, cls.batch_size, air_cell, cls.prior_cell)
        output = model(cls.img, cls.z_tm1, cls.temporal_conditioning, cls.prior_state)
        return air_cell, model, output

    def test_shapes(self):
        num_hidden_outputs = 9
        num_other_outputs = 10

        self.assertEqual(len(self.output.hidden_outputs), num_hidden_outputs)
        self.assertEqual(len(self.output), num_other_outputs + len(self.output.hidden_outputs))
        for k, v in self.output.iteritems():
            if k not in 'hidden_outputs prior_stats prior_state'.split():
                shape = v.shape.as_list()
                self.assertEqual(shape[0], self.batch_size, '{} has incorrect shape={}'.format(k, v.shape))
                if len(shape) > 1:
                    self.assertEqual(shape[1], self.n_steps)

    def test_values(self):

        random_img = np.random.uniform(size=self.img.shape.as_list())
        values, pres_tm1 = self.sess.run([self.output, self.pres_tm1], {self.img: random_img})
        pres_tm1 = pres_tm1[..., 0]

        for k, v in values.iteritems():
            self.assertFalse(flatten_check(v, np.isnan), 'NaNs in {}'.format(k))
            self.assertFalse(flatten_check(v, np.isinf), 'infs in {}'.format(k))

        # KL is estimated by sampling, but mean should be nonnegative
        self.assertGreaterEqual(values.kl.mean(), 0.)
        self.assertGreaterEqual(values.kl_what.mean(), 0.)
        self.assertGreaterEqual(values.kl_where.mean(), 0.)
        self.assertGreaterEqual(values.kl_num_step.mean(), 0.)

        # what & where KL should be zero for those entries where pres_tm1 is zero
        assert_array_equal(values.kl_what, values.kl_what * pres_tm1)
        assert_array_equal(values.kl_where, values.kl_where * pres_tm1)

        # presence should be 0 if presence_tm1 is zero; if presence_tm1 is not zero then presence may or may not be 0
        pres = values.presence
        zero_idx = np.where(1. - pres_tm1)
        assert_array_equal(pres[zero_idx], 0.)

        self.assertLessEqual(pres.sum(), pres_tm1.sum())
        self.assertGreater(pres.sum(), 0.)

    def test_all_zero_presence(self):

        random_img = np.random.uniform(size=self.img.shape.as_list())
        pres_tensor = self.output.presence

        pres_tm1 = np.zeros(pres_tensor.shape.as_list(), dtype=np.float32)
        pres, pres_tm1_output = self.sess.run([pres_tensor, self.pres_tm1],
                                              {self.img: random_img, self.pres_tm1: pres_tm1})

        # just to confirm that pres_tm1 is all zeros
        self.assertEqual(np.count_nonzero(pres_tm1), 0.)
        # confirm that pres_tm1 was actually substituted for the random variable
        self.assertEqual(np.count_nonzero(pres_tm1_output), 0.)
        # new presence should not be introduced
        self.assertEqual(np.count_nonzero(pres), 0.)

    def test_all_ones_presence(self):
        random_img = np.random.uniform(size=self.img.shape.as_list())
        pres_tensor = self.output.presence

        pres_tm1 = np.ones(pres_tensor.shape.as_list(), dtype=np.float32)
        pres = self.sess.run(pres_tensor, {self.img: random_img, self.pres_tm1: pres_tm1})

        self.assertLess(np.count_nonzero(pres), pres.size)


class APDRTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(APDRTest, cls).setUpClass()
        tf.reset_default_graph()

        vars_before = tf.trainable_variables()

        cls.batch_size = 10
        cls.img_size = (5, 7)
        cls.crop_size = (2, 2)
        cls.n_latent = 13
        cls.n_steps = 3
        cls.step_success_prob = .9

        cls.img = tf.placeholder(tf.float32, (cls.batch_size,) + cls.img_size, name='inpt')

        cls.disc_cell, cls.disc_model = cls._make_discovery_model()
        cls.prop_cell, cls.prop_model = cls._make_propagation_model()

        cls.temporal_cell = snt.LSTM(5)
        cls.temporal_state = cls.temporal_cell.initial_state(cls.batch_size, tf.float32, trainable=True)
        cls.temporal_conditioning, _ = cls.temporal_cell(cls.temporal_state[0], cls.temporal_state)

        cls.apdr = APDR(cls.n_steps, cls.batch_size, cls.prop_model, cls.disc_model, cls.temporal_cell)
        cls.init_prior_rnn_state = cls.apdr.prior_init_state()
        cls.prior_rnn_state = tf.tile(tf.expand_dims(cls.init_prior_rnn_state, -2), (1, cls.n_steps, 1))

        cls.num_obj_counts = tf.ones((cls.batch_size, cls.n_steps + 1))
        cls.prop_prior_state = [cls.prior_rnn_state, cls.num_obj_counts]

        cls.what_tm1 = tf.random_normal((cls.batch_size, cls.n_steps, cls.n_latent), name='what_tm1')
        cls.where_tm1 = tf.random_normal((cls.batch_size, cls.n_steps, 4), name='where_tm1')
        cls.pres_tm1 = tf.to_float(
            tf.random_uniform((cls.batch_size, cls.n_steps, 1), 0, 2, tf.int32, name='pres_tm1'))

        mean = 3 * cls.pres_tm1 - 3 * (1. - cls.pres_tm1)
        cls.pres_logit_tm1 = tf.random_normal((cls.batch_size, cls.n_steps, 1)) + mean
        cls.z_tm1 = [cls.what_tm1, cls.where_tm1, cls.pres_tm1, cls.pres_logit_tm1]

        # cls.highest_used_id = -1 * tf.ones((cls.batch_size, 1))
        # cls.prev_ids = -1 * tf.ones((cls.batch_size, cls.n_steps, 1))

        # sample highest_used_ids from a uniform distirbution
        cls.highest_used_id = tf.round(tf.random_uniform((cls.batch_size, 1), maxval=100))
        # for present object, set ids uniformly to somwehre between 0 and highest_used_ids
        prev_ids = tf.random_uniform((cls.batch_size, cls.n_steps, 1))
        # make sure that object ids are unique....
        prev_ids = tf.cumsum(prev_ids, 1)
        prev_ids /= prev_ids[:, -1, tf.newaxis]
        prev_ids = tf.round(prev_ids * (cls.highest_used_id[:, tf.newaxis] - cls.n_steps)) \
                   + tf.range(cls.n_steps, dtype=tf.float32)[tf.newaxis, :, tf.newaxis]

        cls.prev_ids = prev_ids * cls.pres_tm1 - (1 - cls.pres_tm1)

        cls.output = cls.apdr(cls.img, cls.z_tm1, cls.temporal_state, cls.prop_prior_state,
                              cls.highest_used_id, cls.prev_ids)

        cls.sess = tf.Session()
        cls.sess.run(tf.global_variables_initializer())

        cls.model_vars = print_trainable_variables(cls.__name__, vars_before)
        cls.num_model_vars = len(cls.model_vars)

    @classmethod
    def _make_discovery_model(cls):
        cls.disc_modules = make_modules()
        air_cell = AIRCell(cls.img_size, cls.crop_size, cls.n_latent, condition_on_inpt=False, **cls.disc_modules)

        # cls.num_present_objects = tf.zeros((cls.batch_size,), name='num_present_objects')
        cls.num_present_objects = tf.to_float(tf.random_uniform((cls.batch_size,), 0, cls.n_steps + 1,
                                                                tf.int32, name='num_present_objects'))
        cls.conditioning = None

        model = AttendDiscoverRepeat(cls.n_steps, cls.batch_size, air_cell, cls.step_success_prob)
        return air_cell, model

    @classmethod
    def _make_propagation_model(cls):
        cls.prop_modules = make_modules()
        air_cell = PropagatingAIRCell(cls.img_size, cls.crop_size, cls.n_latent, **cls.prop_modules)
        cls.prior_cell = snt.GRU(cls.n_latent)

        model = AttendPropagateRepeat(cls.n_steps, cls.batch_size, air_cell, cls.prior_cell)
        return air_cell, model

    def test_fields(self):
        self.assertEqual(self.apdr._n_steps, self.n_steps)
        self.assertEqual(self.apdr._batch_size, self.batch_size)
        self.assertEqual(self.apdr._propagate, self.prop_model)
        self.assertEqual(self.apdr._discover, self.disc_model)
        self.assertEqual(self.apdr._time_cell, self.temporal_cell)

    def test_shapes(self):
        num_hidden_outputs = 9
        num_other_outputs = 15

        self.assertEqual(len(self.output.hidden_outputs), num_hidden_outputs)
        self.assertEqual(len(self.output), num_other_outputs + len(self.output.hidden_outputs))

        expected_shapes = (self.n_latent, 4, 1, 1)
        for i, (z, es) in enumerate(zip(self.output.z_t, expected_shapes)):
            self.assertEqual(z.shape, (self.batch_size, self.n_steps, es))

        prop_prior_state = self.output.prop_prior_state
        self.assertEqual(prop_prior_state[0].shape, (self.batch_size, self.n_steps, self.n_latent))
        self.assertEqual(prop_prior_state[1].shape, (self.batch_size, self.n_steps + 1))

        outputs = dict(self.output)
        for k in 'prop disc hidden_outputs prop_prior_state z_t temporal_hidden_state'.split():
            del outputs[k]

        for k, v in outputs.iteritems():
            shape = v.shape.as_list()
            self.assertEqual(shape[0], self.batch_size, '{} has incorrect shape={}'.format(k, v.shape))
            if len(shape) > 1:
                self.assertTrue(shape[1] in (1, self.n_steps))

        # z_t is a list and should contain what, where, pres, pres_logit
        self.assertEqual(len(self.output.z_t), 4)
        self.assertEqual(self.output.z_t[-1].shape, self.output.z_t[-2].shape)
        self.assertEqual(int(self.output.z_t[0].shape[-1]), self.n_latent)
        self.assertEqual(int(self.output.z_t[1].shape[-1]), 4)
        self.assertEqual(int(self.output.z_t[2].shape[-1]), 1)
        self.assertEqual(int(self.output.z_t[3].shape[-1]), 1)

    def test_values(self):

        random_img = np.random.uniform(size=self.img.shape.as_list())
        values = self.sess.run(self.output, {self.img: random_img})

        for k, v in values.iteritems():
            self.assertFalse(flatten_check(v, np.isnan), 'NaNs in {}'.format(k))
            self.assertFalse(flatten_check(v, np.isinf), 'infs in {}'.format(k))

        # KL is estimated by sampling so it can be negative, but its mean should be nonnegative
        self.assertGreaterEqual(values.kl.mean(), 0.)
        self.assertGreaterEqual(values.kl_what.mean(), 0.)
        self.assertGreaterEqual(values.kl_where.mean(), 0.)
        self.assertGreaterEqual(values.kl_num_step.mean(), 0.)

    def test_all_zero_presence(self):
        """No objects should be propagated; all ids should be introduced by discovery and different than previous ids"""
        random_img = np.random.uniform(size=self.img.shape.as_list())

        pres_tm1 = np.zeros(self.output.presence.shape.as_list(), dtype=np.float32)
        feed_dict = {self.img: random_img, self.pres_tm1: pres_tm1}
        tensors = [self.output, self.prev_ids, self.init_prior_rnn_state[0]]
        values, prev_ids, init_rnn_state = self.sess.run(tensors, feed_dict)

        prev_ids = prev_ids.squeeze()
        prop_pres = values.prop.presence.squeeze()
        obj_ids = values.obj_ids.squeeze()

        # print prop_pres.sum(), pres_tm1.sum()
        for i in xrange(self.batch_size):
            # print 'prev: {} \tnow: {}'.format(prev_ids[i].squeeze(), values.obj_ids[i].squeeze()),\
            #        prop_pres[i].squeeze()
            for j in xrange(self.n_steps):
                self.assertTrue(obj_ids[i, j] == -1 or obj_ids[i, j] not in prev_ids[i])

        self.assertEqual(prop_pres.sum(), 0.)

        # all prior hidden states should be equal to the initial hidden state, since they are initialised
        # anew for every discovered objects
        pres = values.presence.squeeze()
        prior_rnn_state, obj_counts = values.prop_prior_state
        for i in xrange(self.batch_size):
            for j in xrange(self.n_steps):
                if pres[i, j]:
                    assert_array_equal(prior_rnn_state[i, j], init_rnn_state)

    def test_all_ones_presence(self):
        """Some objects should be propagated and all propagated objects should have the same ids"""
        random_img = np.random.uniform(size=self.img.shape.as_list())

        pres_tm1 = np.ones(self.output.presence.shape.as_list(), dtype=np.float32)
        feed_dict = {self.img: random_img, self.pres_tm1: pres_tm1}
        tensors = [self.output, self.prev_ids, self.init_prior_rnn_state[0]]
        values, prev_ids, init_rnn_state = self.sess.run(tensors, feed_dict)

        prev_ids = prev_ids.squeeze()
        prop_pres = values.prop.presence.squeeze()
        obj_ids = values.obj_ids.squeeze()

        # print prop_pres.sum(), pres_tm1.sum()
        for i in xrange(self.batch_size):
            # print 'prev: {} \tnow: {}'.format(prev_ids[i].squeeze(), values.obj_ids[i].squeeze()), prop_pres[
            #     i].squeeze()
            for j in xrange(self.n_steps):
                if prop_pres[i, j]:
                    self.assertTrue(prev_ids[i, j] in obj_ids[i])
                else:
                    self.assertFalse(prev_ids[i, j] in obj_ids[i])

        self.assertLessEqual(prop_pres.sum(), pres_tm1.sum())

        # for every propagated object, the prior hidden state should be different from the
        # initial state and propagated prior states should be different from each other;
        # since present states are pushed to the beginning, we can take the first num_prop
        # states to check
        prop_num_step = values.prop.num_steps.squeeze()
        # disc_num_step = values.disc.num_steps.squeeze()
        prior_rnn_state = values.prop_prior_state[0]
        total_objs = values.num_steps.squeeze()

        for i in xrange(self.batch_size):
            for j in xrange(prop_num_step[i]):
                # print i, j
                # print prior_rnn_state[i, j, :5], init_rnn_state[:5]
                # different than the initial state
                self.assertFalse((prior_rnn_state[i, j] == init_rnn_state).all())
                # different than states for previous objects at this timesteps
                for k in xrange(j):
                    self.assertFalse((prior_rnn_state[i, j] == prior_rnn_state[i, k]).all())

            # here we check until prop_num_step[i] + disc_num_step[i], because some states might
            # be populated by propagated states for which prop_presence==0
            # print 'end prop'

            # for j in xrange(prop_num_step[i], self.n_steps):
            for j in xrange(prop_num_step[i], total_objs[i]):
                # print i, j
                # print prior_rnn_state[i, j, :5], init_rnn_state[:5]
                assert_array_equal(prior_rnn_state[i, j], init_rnn_state)
                # print 'end disc'