# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

import numpy as np
import paddle
from paddleaudio.functional.window import get_window

from .base import FeatTest
from paddlespeech.audio.transform.spectrogram import Stft


class TestStft(FeatTest):
    def initParmas(self):
        self.n_fft = 512
        self.hop_length = 128
        self.window_str = 'hann'

    def test_stft(self):
        ps_stft = Stft(self.n_fft, self.hop_length)
        ps_res = ps_stft(
            self.waveform.T).squeeze(1).T  # (n_fft//2 + 1, n_frmaes)

        x = paddle.to_tensor(self.waveform)
        window = get_window(self.window_str, self.n_fft, dtype=x.dtype)
        pd_res = paddle.signal.stft(
            x, self.n_fft, self.hop_length, window=window).squeeze(0).numpy()

        np.testing.assert_array_almost_equal(ps_res, pd_res, decimal=5)


if __name__ == '__main__':
    unittest.main()
