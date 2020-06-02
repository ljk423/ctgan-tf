import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os
from unittest import TestCase

from ctgan.utils import generate_data, get_test_variables
from ctgan.synthesizer import CTGANSynthesizer


class TestSynthesizer(TestCase):
    def setUp(self):
        self._vars = get_test_variables()
        self._n_samples = 1
        self._current_dir = os.path.dirname(os.path.abspath(__file__))
        self._expected_values = joblib.load(os.path.join(
            self._current_dir, 'expected_values.joblib'))

    def tearDown(self):
        model_path = os.path.join(self._current_dir, 'model_test.joblib')
        if os.path.exists(model_path):
            os.remove(model_path)

        del self._vars
        del self._n_samples
        del self._current_dir
        del self._expected_values

    def _assert_train_equal(self, data, discrete):
        model = CTGANSynthesizer(
            batch_size=self._vars['batch_size'], pac=self._vars['pac'])
        self.assertIsNotNone(model)
        model.train(data, discrete, epochs=1)
        outputs = {
            'output_tensor': [x.numpy()
                              for x in model._transformer.output_tensor],
            'cond_tensor': [x.numpy() for x in model._transformer.cond_tensor],
            'gen_weights':  model._generator.get_weights(),
            'crt_weights': model._critic.get_weights(),
        }

        idx = int(len(discrete) > 0)
        for o in outputs:
            for i in range(len(outputs[o])):
                np.testing.assert_almost_equal(
                    outputs[o][i], self._expected_values[idx][o][i],
                    decimal=self._vars['decimal'])

    def test_train(self):
        np.random.seed(0)
        tf.random.set_seed(0)
        data, discrete = generate_data(self._vars['batch_size'])
        self._assert_train_equal(data, [])
        self._assert_train_equal(data, discrete)

    def test_sample(self):
        np.random.seed(0)
        tf.random.set_seed(0)
        data, discrete = generate_data(self._vars['batch_size'])

        model = CTGANSynthesizer(
            batch_size=self._vars['batch_size'], pac=self._vars['pac'])
        self.assertIsNotNone(model)

        model.train(data, discrete, epochs=1)
        output = model.sample(self._n_samples).values
        expected_output = np.array([[0.4139329, 3.0]])
        np.testing.assert_almost_equal(
            output, expected_output, decimal=self._vars['decimal'])

    def test_model_to_disk(self):
        np.random.seed(0)
        tf.random.set_seed(0)
        data, discrete = generate_data(self._vars['batch_size'])

        model = CTGANSynthesizer(
            batch_size=self._vars['batch_size'], pac=self._vars['pac'])
        self.assertIsNotNone(model)
        model.train(data, discrete, epochs=1)
        model_path = os.path.join(self._current_dir, 'model_test.joblib')
        model.dump(model_path, overwrite=True)
        loaded_model = CTGANSynthesizer(file_path=model_path)
        self.assertIsNotNone(loaded_model)

        for attr, value in loaded_model.__dict__.items():
            self.assertTrue(attr in model.__dict__)
            if type(value) in [int, float, tuple]:
                self.assertEqual(value, model.__dict__[attr])

        np.testing.assert_equal(
            loaded_model._cond_generator.__dict__,
            model._cond_generator.__dict__)

        for attr, value in loaded_model._transformer.__dict__.items():
            if isinstance(value, pd.Series):
                pd.testing.assert_series_equal(
                    value, model._transformer.__dict__[attr])
            elif isinstance(value, list) and isinstance(value[0], tf.Tensor):
                tf.assert_equal(value, model._transformer.__dict__[attr])
            else:
                np.testing.assert_equal(
                    value, model._transformer.__dict__[attr])

        np.testing.assert_equal(
            loaded_model._generator.get_weights(),
            model._generator.get_weights())
