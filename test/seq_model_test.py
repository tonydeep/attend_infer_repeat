import unittest
import time
import tensorflow as tf
import sonnet as snt

from attrdict import AttrDict

from attend_infer_repeat.seq_model import SeqAIRModel
from attend_infer_repeat.elbo import AIRPriorMixin, KLMixin, LogLikelihoodMixin
from attend_infer_repeat.modules import *
from attend_infer_repeat.seq_mixins import NaiveSeqAirMixin, SeparateSeqAIRMixin


class AIRModelWithPriors(SeqAIRModel, AIRPriorMixin, KLMixin, LogLikelihoodMixin, SeparateSeqAIRMixin):
# class AIRModelWithPriors(SeqAIRModel, AIRPriorMixin, KLMixin, LogLikelihoodMixin, NaiveSeqAirMixin):
    importance_resample = True
    pass


def make_modules():

    return dict(
        transition=snt.GRU(3),
        input_encoder=(lambda: Encoder(5)),
        glimpse_encoder=(lambda: Encoder(7)),
        glimpse_decoder=(lambda x: Decoder(11, x)),
        transform_estimator=(lambda: StochasticTransformParam(13)),
        steps_predictor=(lambda: StepsPredictor(17))
    )


class SeqModelTest(unittest.TestCase):
    learning_rate = 1e-4
    batch_size = 10
    img_size = (5, 7)
    crop_size = (2, 2)
    n_what = 13
    n_steps_per_image = 3
    iw_samples = 2
    n_timesteps = 2

    @classmethod
    def setUpClass(cls):
        cls.timer_dict = dict()

        cls.imgs = tf.placeholder(tf.float32, (cls.n_timesteps, cls.batch_size,) + cls.img_size, name='inpt')
        cls.nums = tf.placeholder(tf.float32, (cls.n_timesteps, cls.batch_size, cls.n_steps_per_image), name='nums')

        print 'Building AIR'
        cls.modules = make_modules()
        time_start = time.clock()
        cls.air = AIRModelWithPriors(cls.imgs, cls.n_steps_per_image, cls.crop_size, cls.n_what,
                                     condition_on_prev=True,
                                     condition_on_latents=True,
                                     prior_around_prev=True,
                                     iw_samples=cls.iw_samples, **cls.modules)
        cls.rnn_outputs = AttrDict({k: getattr(cls.air, k) for k in cls.air.cell.output_names})
        cls.outputs = AttrDict({k: getattr(cls.air, k) for k in cls.air.output_names})

        print 'Constructed model'

        cls.train_step = cls.air.train_step(cls.learning_rate, nums=cls.nums)
        cls.loss = tf.reduce_mean(cls.air.iw_elbo)
        cls.loss = cls.air.nelbo / cls.air.n_timesteps
        print 'Computed gradients'
        cls.register_time(time_start, time.clock(), 'Building model')

    @classmethod
    def tearDownClass(cls):
        for k, v in cls.timer_dict.iteritems():
            print '{} took {}s'.format(k, v)
        super(SeqModelTest, cls).tearDownClass()

    @classmethod
    def register_time(cls, start_time, end_time, name):
        cls.timer_dict[name] = end_time - start_time

    def test_forward(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        xx = np.random.rand(*self.imgs.get_shape().as_list())

        time_start = time.clock()

        rnn_outputs, outputs, l = sess.run([self.rnn_outputs, self.outputs, self.loss], {self.imgs: xx})

        self.register_time(time_start, time.clock(), 'Forward pass')
        print 'rnn_outputs:'
        for k, v in rnn_outputs.iteritems():
            print k, v.shape

        print
        print 'outputs:'
        for k, v in outputs.iteritems():
            print k, v.shape

        print
        print 'loss = {}'.format(l)
        self.assertLess(l, 70.)
        self.assertGreater(l, 39.)

        print 'obj_ids'
        for i in xrange(self.batch_size):
            print
            for t in xrange(self.n_timesteps):
                print 'i={}, t={}, id={}, p={}'.format(i, t, outputs.obj_id[t, i].squeeze(), rnn_outputs.presence[t, i].squeeze())

    def test_backward(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        xx = np.random.rand(*self.imgs.get_shape().as_list())

        print 'Running train step'

        time_start = time.clock()

        sess.run(self.train_step, {self.imgs: xx})
        self.register_time(time_start, time.clock(), 'Backward pass')

        print 'Done'

    def test_shapes(self):
        learning_signal_shape = self.air.num_steps_learning_signal.shape.as_list()
        self.assertEqual(learning_signal_shape, [self.batch_size, 1])



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
