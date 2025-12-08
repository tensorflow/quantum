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

"""Tests for clean_distribution script."""

import os
import subprocess
import unittest


class CleanDistributionTest(unittest.TestCase):
    """Tests for clean_distribution script."""

    def setUp(self):
        # Find the repo root
        try:
            self.repo_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                universal_newlines=True).strip()
        except subprocess.CalledProcessError:
            self.repo_root = os.getcwd()

        self.script = os.path.join(self.repo_root, "release",
                                   "clean_distribution")

    def test_dry_run(self):
        """Test clean_distribution script in dry-run mode."""
        wheel_path = "fake.whl"
        cmd = [self.script, "-n", "-p", "3.10", wheel_path]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=False)
        self.assertEqual(result.returncode, 0,
                         f"Script failed with stderr: {result.stderr}")
        output = result.stdout

        self.assertIn("(Dry run) docker run", output)
        self.assertIn("auditwheel repair", output)
        self.assertIn(f"/tmp/{wheel_path}", output)

    def test_show_action(self):
        """Test clean_distribution script with -s (show) action."""
        wheel_path = "fake.whl"
        cmd = [self.script, "-n", "-s", wheel_path]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=False)
        self.assertEqual(result.returncode, 0,
                         f"Script failed with stderr: {result.stderr}")
        output = result.stdout

        self.assertIn("auditwheel show", output)

    def test_missing_arg(self):
        """Test clean_distribution script fails without wheel argument."""
        cmd = [self.script, "-n"]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ERROR: insufficient arguments",
                      result.stdout + result.stderr)

    def test_help(self):
        """Test clean_distribution script help flag."""
        cmd = [self.script, "-h"]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=False)
        self.assertEqual(result.returncode, 0,
                         f"Script failed with stderr: {result.stderr}")
        self.assertIn("Usage:", result.stdout)
        self.assertIn("Run auditwheel on the given wheel file.", result.stdout)

    def test_invalid_option(self):
        """Test clean_distribution script with invalid option."""
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
