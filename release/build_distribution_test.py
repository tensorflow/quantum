# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for build_distribution script."""

import os
import subprocess
import unittest


class BuildDistributionTest(unittest.TestCase):
    """Tests for build_distribution script."""

    def setUp(self):
        # Find the repo root. Default to the current directory if that fails.
        try:
            self.repo_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                universal_newlines=True).strip()
        except subprocess.CalledProcessError:
            self.repo_root = os.getcwd()

        self.script = os.path.join(self.repo_root, "release",
                                   "build_distribution")

    def test_dry_run(self):
        """Test build_distribution script in dry-run mode."""
        cmd = [self.script, "-n", "-c", "11.2", "-p", "3.9", "-t", "2.10"]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=False)
        self.assertEqual(result.returncode, 0,
                         f"Script failed with stderr: {result.stderr}")
        output = result.stdout

        self.assertIn("(Dry run) docker run", output)
        # Check that arguments propagated to the docker image tag and env vars
        self.assertIn("tensorflow/build:2.10-python3.9", output)
        self.assertIn("cuda_version=11.2", output)
        self.assertIn("py_version=3.9", output)
        self.assertIn("tf_version=2.10", output)

        self.assertIn(
            "(Dry run) check-wheel-contents /tmp/tensorflow_quantum/*.whl",
            output)

    def test_defaults(self):
        """Test build_distribution script defaults."""
        cmd = [self.script, "-n"]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=False)
        self.assertEqual(result.returncode, 0,
                         f"Script failed with stderr: {result.stderr}")
        output = result.stdout
        # Check default cuda version (12) and default cleanup (true)
        self.assertIn("cuda_version=12", output)
        self.assertIn("cleanup=true", output)

    def test_help(self):
        """Test build_distribution script help flag."""
        cmd = [self.script, "-h"]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=False)
        self.assertEqual(result.returncode, 0,
                         f"Script failed with stderr: {result.stderr}")
        self.assertIn("Usage:", result.stdout)
        self.assertIn("Build a distribution wheel for TensorFlow Quantum.",
                      result.stdout)

    def test_invalid_option(self):
        """Test build_distribution script with invalid option."""
        cmd = [self.script, "-z"]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Usage:", result.stdout + result.stderr)
        stderr = result.stderr.lower()
        self.assertTrue(
            "illegal option" in stderr or "invalid option" in stderr,
            "Expected 'illegal option' or 'invalid option' in stderr, "
            f"got: {stderr}"
        )


if __name__ == "__main__":
    unittest.main()
