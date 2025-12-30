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

import os
import glob
import re

from absl.testing import parameterized
import nbformat
import nbclient

# Ensure we always use legacy tf.keras (Keras 2) when running tutorials.
# This must be set before importing TensorFlow so it picks up tf_keras.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Pylint doesn't like code before imports, but we need the env var set first.
import tensorflow as tf  # pylint: disable=wrong-import-position

# Must be run from the directory containing `quantum` repo.
NOTEBOOKS = glob.glob("quantum/docs/tutorials/*.ipynb")


class ExamplesTest(tf.test.TestCase, parameterized.TestCase):
    """Execute all tutorial notebooks and check they run without errors."""

    @parameterized.parameters(NOTEBOOKS)
    def test_notebook(self, path):
        """Test that notebooks open and run correctly."""

        nb = nbformat.read(path, as_version=4)
        # Scrub any magic from the notebook before running.
        for cell in nb.get("cells"):
            if cell["cell_type"] == "code":
                src = cell["source"]
                # Comment out lines with '!' or '%' (typically magic commands)
                # but not '!='. Preserve indentation to avoid syntax errors.
                lines = src.split('\n')
                new_lines = []
                for line in lines:
                    if re.match(r"^\s*(\!|%)(?!=)", line):
                        # Replace ! or % with 'pass # !' or 'pass # %'
                        match = re.search(r"(\!|%)", line)
                        idx = match.start()
                        new_lines.append(line[:idx] + "pass # " + line[idx:])
                    else:
                        new_lines.append(line)
                src = '\n'.join(new_lines)

                # For mnist.ipynb to reduce runtime in test.
                src = re.sub(r"NUM_EXAMPLES ?= ?.*", "NUM_EXAMPLES = 10", src)
                # For quantum_reinforcement_learning.ipynb:
                # reduce runtime in test by limiting episodes.
                src = re.sub(r"n_episodes ?= ?.*", "n_episodes = 50", src)
                # For noise.ipynb to reduce runtime in test.
                src = re.sub(r"n_epochs ?= ?.*", "n_epochs = 2", src)
                cell["source"] = src

        _ = nbclient.execute(nb, timeout=900, kernel_name="python3")


if __name__ == "__main__":
    tf.test.main()
