import numpy as np
from unittest import TestCase

from ctgan.utils import generate_data
from ctgan.data_modules import ConditionalGenerator, DataTransformer


class TestConditionalGenerator(TestCase):
    def setUp(self):
        self._input_dim = 10
        self._pac = 10
        self._batch_size = 10

    def tearDown(self):
        del self._input_dim
        del self._pac
        del self._batch_size

    def test_sample(self):
        np.random.seed(0)
        data, discrete = generate_data(self._batch_size)

        transformer = DataTransformer()
        transformer.fit(data, discrete)
        train_data = transformer.transform(data)

        cond_gen = ConditionalGenerator(
            train_data, transformer.output_info, True)
        output = cond_gen.sample(self._batch_size)
        self.assertIsNotNone(output)
        c, m, col, opt = output
        expected_c = np.array([
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
            [0., 1., 0., 0.]], dtype=np.float32)
        expected_m = np.array([[1.],
                               [1.],
                               [1.],
                               [1.],
                               [1.],
                               [1.],
                               [1.],
                               [1.],
                               [1.],
                               [1.]], dtype=np.float32)
        expected_col = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        expected_opt = np.array([0, 2, 0, 0, 0, 0, 0, 1, 3, 1])
        np.testing.assert_equal(c, expected_c)
        np.testing.assert_equal(m, expected_m)
        np.testing.assert_equal(col, expected_col)
        np.testing.assert_equal(opt, expected_opt)

        output = cond_gen.sample_zero(self._batch_size)
        self.assertIsNotNone(output)
        print(output)
        expected_output = [
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.]]
        np.testing.assert_equal(output, expected_output)

    def test_sample_none(self):
        np.random.seed(0)
        data, discrete = generate_data(self._batch_size)

        transformer = DataTransformer()
        transformer.fit(data, [])
        train_data = transformer.transform(data)

        cond_gen = ConditionalGenerator(
            train_data, transformer.output_info, True)
        output = cond_gen.sample(self._batch_size)
        self.assertIsNone(output)

        output = cond_gen.sample_zero(self._batch_size)
        self.assertIsNone(output)
