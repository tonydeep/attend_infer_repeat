import tensorflow as tf
import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal
from testing_tools import TFTestBase

from attend_infer_repeat.ops import sample_from_tensor, gather_axis, tile_input_for_iwae, select_present, compute_object_ids


class SampleFromTensorTest(TFTestBase):

    @classmethod
    def setUpClass(cls):
        super(SampleFromTensorTest, cls).setUpClass()
        cls.n_dim = 3

        for i in xrange(1, 4):
            d = '{}d'.format(i)
            local = {
                'x'+d: tf.placeholder(tf.float32, [None] * (i-1) + [cls.n_dim], 'x' + d),
                'y'+d: tf.placeholder(tf.float32, [None] * (i-1), 'y' + d),
            }

            local['sample'+d] = sample_from_tensor(local['x'+d], local['y'+d])

            for k, v in local.iteritems():
                setattr(cls, k, v)

    def test_sample1d(self):
        """single vectors"""

        x = [5, 7, 11]
        for y in xrange(3):
            r = self.eval(self.sample1d, feed_dict={self.x1d: x, self.y1d: y})

            self.assertEqual(r, x[y])

    def test_sample2d(self):
        """minibatches of observations"""

        x = np.asarray([[5, 7, 11]])
        x = np.tile(x, (4, 1)) + np.arange(4)[:, np.newaxis]
        r = self.eval(self.sample2d, feed_dict={self.x2d: x, self.y2d: [0, 1, 2, 0]})
        assert_array_equal(r, [5, 8, 13, 8])

    def test_sample3d(self):
        """minibatches of timeseries"""

        x = np.asarray([[[5, 7, 11]]])
        x = np.tile(x, (2, 4, 1)) + np.arange(8).reshape(2, 4, 1)
        r = self.eval(self.sample3d, feed_dict={self.x3d: x, self.y3d: [[2, 1, 1, 2], [0, 1, 2, 0]]})
        assert_array_equal(r, [[11, 8, 9, 14], [9, 12, 17, 12]])


class GatherAxisTest(TFTestBase):
    vars = {
        'x': [tf.float32, None],
        'y': [tf.int32, None],
        'm': [tf.int32, None]
    }

    @classmethod
    def setUpClass(cls):
        super(GatherAxisTest, cls).setUpClass()
        cls.gather = gather_axis(cls.x, cls.y, cls.m)

    def test_axis0(self):

        tensor = np.arange(10) + 1
        idx = [3, 4, 5]
        res = self.eval(self.gather, tensor, idx, 0)
        self.assertEqual(res.shape, (3,))
        assert_array_equal(res, [4, 5, 6])

    def test_axis1_last(self):

        tensor = np.arange(10).reshape(1, 10) + 1
        idx = [3, 4, 5]
        expected = tensor[:, idx]

        res = self.eval(self.gather, tensor, idx, 1)
        self.assertEqual(res.shape, expected.shape)
        assert_array_equal(res, expected)

    def test_axis1_not_last(self):

        tensor = np.arange(30).reshape(1, 10, 3) + 1
        idx = [3, 5, 7]
        expected = tensor[:, idx]

        res = self.eval(self.gather, tensor, idx, 1)

        self.assertEqual(res.shape, expected.shape)
        assert_array_equal(res, expected)

    def test_axis1_first_not_equal_to_one(self):

        tensor = np.arange(30).reshape(3, 10) + 1
        idx = [3, 5, 7]
        expected = tensor[:, idx]

        res = self.eval(self.gather, tensor, idx, 1)
        self.assertEqual(res.shape, expected.shape)
        assert_array_equal(res, expected)

    def test_complex(self):
        tensor = np.arange(210).reshape(3, 10, 7) + 1
        idx = [3, 5, 7]
        expected = tensor[:, idx]

        res = self.eval(self.gather, tensor, idx, 1)
        self.assertEqual(res.shape, expected.shape)
        assert_array_equal(res, expected)

    def test_neg_axis(self):
        tensor = np.arange(210).reshape(3, 10, 7) + 1
        idx = [3, 5, 7]
        expected = tensor[:, idx]

        res = self.eval(self.gather, tensor, idx, -2)
        self.assertEqual(res.shape, expected.shape)
        assert_array_equal(res, expected)


class TileInputForIwaeTest(TFTestBase):

    vars = {
        'x': [tf.float32, [5, 1]],
        'y': [tf.int32, None]
    }

    @classmethod
    def setUpClass(cls):
        super(TileInputForIwaeTest, cls).setUpClass()
        cls.tile = tile_input_for_iwae(cls.x, cls.y, False)

        x = cls.x[tf.newaxis]
        cls.tile_timed = tile_input_for_iwae(x, cls.y, True)

    def test_no_time(self):

        batch = np.random.randn(5, 1)
        tiled = self.eval(self.tile, batch, 7)

        self.assertEqual(tiled.shape, (35, 1))
        tiled = tiled.reshape(5, 7, 1)
        for i in xrange(7):
            assert_array_almost_equal(tiled[:, i], batch)

    def test_time(self):

        batch = np.random.randn(1, 5, 1)
        tiled = self.eval(self.tile_timed, batch[0], 7)
        self.assertEqual(tiled.shape, (1, 35, 1))
        tiled = tiled.reshape(1, 5, 7, 1)
        for i in xrange(7):
            assert_array_almost_equal(tiled[:, :, i], batch)


class TestSelectPresent(TFTestBase):
    def setUp(self):
        self.sp = select_present(self.x, self.y)
        self.spm = lambda bs: select_present(self.x, self.y, bs)

    def test_batch_1d(self):
        x = np.asarray([[1, 2], [3, 4]])
        m = np.asarray([0, 1])
        yy = np.asarray([[3, 4], [1, 2]])
        y = self.eval(self.sp, x, m)
        assert_array_equal(y, yy)

    def test_timed_batch_1d(self):
        x = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        m = np.asarray([[0, 1], [1, 0]])
        yy = np.asarray([[[3, 4], [1, 2]], [[5, 6], [7, 8]]])

        y = self.eval(self.spm(2), x, m)
        assert_array_equal(y, yy)

    def test_batch_2d(self):
        x = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        m = np.asarray([0, 1])
        yy = np.asarray([[[5, 6], [7, 8]], [[1, 2], [3, 4]]])

        y = self.eval(self.sp, x, m)
        assert_array_equal(y, yy)


class ComputeObjectIdsTest(TFTestBase):

    def test_no_obj(self):
        last_used_id = np.asarray([-1]).reshape(1, 1)
        prev_ids = np.asarray([-1]*3).reshape(1, 3, 1)
        prop_pres = np.asarray([0]*3).reshape(1, 3, 1)
        disc_pres = np.asarray([0]*3).reshape(1, 3, 1)

        res = list(compute_object_ids(last_used_id, prev_ids, prop_pres, disc_pres))
        for i, r in enumerate(res):
            res[i] = self.eval(r)
        used, new = res

        assert_array_equal(used, last_used_id)
        assert_array_equal(new[:, :3], prev_ids)
        assert_array_equal(new[:, 3:], prev_ids)

    def test_prop_obj(self):
        last_used_id = np.asarray([3]).reshape(1, 1)
        prev_ids = np.asarray([1, -1, 3]).reshape(1, 3, 1)
        prop_pres = np.asarray([1, 0, 0]).reshape(1, 3, 1)
        disc_pres = np.asarray([0]*3).reshape(1, 3, 1)

        res = list(compute_object_ids(last_used_id, prev_ids, prop_pres, disc_pres))
        for i, r in enumerate(res):
            res[i] = self.eval(r)
        used, new = res

        expected_prop = np.asarray([1, -1, -1]).reshape(1, 3, 1)
        expected_disc = np.asarray([-1, -1, -1]).reshape(1, 3, 1)

        assert_array_equal(used, last_used_id)
        assert_array_equal(new[:, :3], expected_prop)
        assert_array_equal(new[:, 3:], expected_disc)

    def test_disc_obj(self):
        last_used_id = np.asarray([3]).reshape(1, 1)
        prev_ids = np.asarray([1, -1, 3]).reshape(1, 3, 1)
        prop_pres = np.asarray([1, 0, 0]).reshape(1, 3, 1)
        disc_pres = np.asarray([1, 1, 0]).reshape(1, 3, 1)

        res = list(compute_object_ids(last_used_id, prev_ids, prop_pres, disc_pres))
        for i, r in enumerate(res):
            res[i] = self.eval(r)
        used, new = res

        expected_last_used_id = np.asarray([5]).reshape(1, 1)
        expected_prop = np.asarray([1, -1, -1]).reshape(1, 3, 1)
        expected_disc = np.asarray([4, 5, -1]).reshape(1, 3, 1)

        assert_array_equal(used, expected_last_used_id)
        assert_array_equal(new[:, :3], expected_prop)
        assert_array_equal(new[:, 3:], expected_disc)
