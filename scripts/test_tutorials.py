# Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.
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
# ==============================================================================
"""Module to ensure all notebooks execute without error by pytesting them."""
import glob
import re

from absl.testing import parameterized
import nbformat
import nbconvert
import tensorflow as tf

# Must be run from the directory containing `quantum` repo.
NOTEBOOKS = glob.glob("quantum/docs/tutorials/*.ipynb")


class ExamplesTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters(NOTEBOOKS)
    def test_notebook(self, path):
        """Test that notebooks open/run correctly."""

        nb = nbformat.read(path, as_version=4)
        # Scrub any magic from the notebook before running.
        for cell in nb.get("cells"):
            if cell['cell_type'] == 'code':
                src = cell['source']
                # Comment out lines containing '!' but not '!='
                src = re.sub(r'\!(?!=)', r'#!', src)
                # For mnist.ipynb to reduce runtime in test.
                src = re.sub('NUM_EXAMPLES ?= ?.*', 'NUM_EXAMPLES = 10', src)
                # For quantum_reinforcement_learning.ipynb to reduce runtime in test.
                src = re.sub('n_episodes ?= ?.*', 'n_episodes = 50', src)
                # For noise.ipynb to reduce runtime in test.
                src = re.sub('n_epochs ?= ?.*', 'n_epochs = 2', src)
                cell['source'] = src

        _ = nbconvert.preprocessors.execute.executenb(nb,
                                                      timeout=900,
                                                      kernel_name="python3")


if __name__ == "__main__":
    tf.test.main()
