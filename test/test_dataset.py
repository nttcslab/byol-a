import sys
import unittest
from byol_a.common import *
from byol_a.dataset import *
from utils.downstream_tasks import create_data_source


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.cfg = load_yaml_config('config.yaml')
        self.files = create_data_source('us8k').subset([0]).files
        self.cfg.unit_sec = 4.0

    def tearDown(self):
        pass

    def test_WaveInLMSOutDataset(self):
        # Preparation: Check the length, and get the entire average of the first file.
        self.cfg.unit_sec = 4.0
        ds = WaveInLMSOutDataset(self.cfg, self.files, labels=None, tfms=None)
        org_time_frames = ds[0].shape[-1] # last dim is the time.
        assert org_time_frames == 401
        base_mean = ds[0].mean().numpy()

        # https://github.com/nttcslab/byol-a/issues/9
        # Test with 1.0 s audio segments.
        # We expect the average of the random 1s sample of the same audio file will
        # converges to the average of entire audio file. We use the first file.
        # Procedure:
        # 1. We repeat getting 1s random sample to calculate moving average.
        # 2. Test if the moving average reaches to the average of entire audio sample.
        self.cfg.unit_sec = 1.0
        ds = WaveInLMSOutDataset(self.cfg, self.files, labels=None, tfms=None)
        assert ds[0].shape[-1] is 101
        cur_mean = 0.0
        for _ in range(10000):
            cur_mean = 0.99 * cur_mean + 0.01 * ds[0].mean().numpy() # moving average
            if np.isclose(cur_mean, base_mean, atol=1e-3):
                break
        self.assertAlmostEqual(cur_mean, base_mean, places=2)
        # Possible failures:
        # Random cropping is not actually `rundom`.


if __name__ == '__main__':
    unittest.main()
