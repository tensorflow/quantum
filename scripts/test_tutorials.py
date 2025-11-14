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
import os
import time
import unittest

from absl.testing import parameterized
import nbclient  # pylint: disable=import-error
import nbformat
from nbformat.v4 import new_code_cell


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DEFAULT_TUTORIAL_ROOT = "quantum/docs/tutorials"
DEFAULT_KERNEL = os.environ.get("NB_KERNEL_NAME", "python3")
CELL_TIMEOUT_SEC = int(os.environ.get("NB_CELL_TIMEOUT", "900"))


def _discover_tutorials(root=DEFAULT_TUTORIAL_ROOT):
    """Return a sorted list of *.ipynb under the tutorials folder."""
    paths = sorted(
        glob.glob(os.path.join(root, "**", "*.ipynb"), recursive=True))
    # Skip checkpoints and hidden files.
    clean = []
    for nb_path in paths:
        base = os.path.basename(nb_path)
        if ".ipynb_checkpoints" in nb_path:
            continue
        if base.startswith("."):
            continue
        clean.append(nb_path)
    return clean


TUTORIAL_PATHS = _discover_tutorials()


def _gym_compat_cell():
    """Return a code cell that shims Gym>=0.26 to old API shape."""
    shim = (
        "import os\n"
        "os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')\n"
        "\n"
        "try:\n"
        "    import gym\n"
        "except Exception:  # pragma: no cover\n"
        "    gym = None\n"
        "\n"
        "if gym is not None:\n"
        "    import types\n"
        "\n"
        "    def _unwrap_reset(res):\n"
        "        if isinstance(res, tuple) and len(res) == 2:\n"
        "            return res[0]\n"
        "        return res\n"
        "\n"
        "    def _unwrap_step(res):\n"
        "        if isinstance(res, tuple) and len(res) == 5:\n"
        "            obs, reward, terminated, truncated, info = res\n"
        "            done = bool(terminated) or bool(truncated)\n"
        "            return obs, reward, done, info\n"
        "        return res\n"
        "\n"
        "    def _wrap_env(env):\n"
        "        if not hasattr(env, '_tfq_wrapped'):\n"
        "            env._orig_reset = env.reset\n"
        "            env._orig_step = env.step\n"
        "            env.reset = types.MethodType(\n"
        "                lambda self: _unwrap_reset(self._orig_reset()), env\n"
        "            )\n"
        "            env.step = types.MethodType(\n"
        "                lambda self, a: _unwrap_step(self._orig_step(a)),\n"
        "                env\n"
        "            )\n"
        "            env._tfq_wrapped = True\n"
        "        return env\n"
        "\n"
        "    if hasattr(gym, 'make'):\n"
        "        _orig_make = gym.make\n"
        "\n"
        "        def _make(name, *args, **kwargs):\n"
        "            return _wrap_env(_orig_make(name, *args, **kwargs))\n"
        "\n"
        "        gym.make = _make\n")
    return new_code_cell(shim)


class ExamplesTest(parameterized.TestCase):
    """Parameterized unittest that executes each discovered notebook."""

    @parameterized.parameters([(p,) for p in TUTORIAL_PATHS])
    def test_notebook(self, nb_path):
        """Execute a single notebook with nbclient."""
        # Load notebook.
        with open(nb_path, "r", encoding="utf-8") as handle:
            nb = nbformat.read(handle, as_version=4)

        # Insert shim as first cell.
        nb.cells.insert(0, _gym_compat_cell())

        # Set working directory for relative paths in the notebook.
        resources = {"metadata": {"path": os.path.dirname(nb_path)}}

        # Log start for visibility similar to GTest output.
        print(f"[ RUN      ] ExamplesTest.test_notebook ('{nb_path}')")
        start = time.time()

        try:
            nbclient.NotebookClient(
                nb=nb,
                kernel_name=DEFAULT_KERNEL,
                timeout=CELL_TIMEOUT_SEC,
                resources=resources,
                allow_errors=False,
            ).execute()
        except nbclient.exceptions.CellTimeoutError as err:
            # Re-raise as a standard error to avoid constructor signature
            # requirements on nbclient's exception types.
            raise RuntimeError(f"Notebook timed out: {nb_path}") from err
        except nbclient.exceptions.CellExecutionError as err:
            raise RuntimeError(f"Execution error in: {nb_path}\n{err}") from err

        dur = time.time() - start
        print("[       OK ] "
              f"ExamplesTest.test_notebook ('{nb_path}') "
              f"({dur:.2f}s)")


if __name__ == "__main__":
    # Print discovered notebooks for visibility in CI logs.
    print("Discovered notebooks:")
    if not TUTORIAL_PATHS:
        print("  (none found)")
    else:
        for nbp in TUTORIAL_PATHS:
            print("  -", nbp)
