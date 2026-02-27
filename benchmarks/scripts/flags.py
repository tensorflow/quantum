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
"""Command line flags shared between benchmarks."""
from collections import namedtuple
from absl import flags as absl_flags

FLAGS = absl_flags.FLAGS

absl_flags.DEFINE_integer('n_qubits',
                          None,
                          'Number of qubits in the benchmark circuit.',
                          lower_bound=2,
                          upper_bound=16)

absl_flags.DEFINE_integer('n_moments',
                          None,
                          'Depth of benchmark circuit.',
                          lower_bound=1)

absl_flags.DEFINE_float(
    'op_density',
    0.99,
    'Density of operators in benchmark circuit, or the probability that a '
    'given qubit in each moment is acted on by an operation.',
    lower_bound=0,
    upper_bound=0.99)  # For compatibility with util lib

absl_flags.DEFINE_integer('n_symbols',
                          1, 'Number of symbols to parametrize a circuit by. '
                          'Use this to tune optimization convergence times.',
                          lower_bound=1)

absl_flags.DEFINE_integer(
    'n_rows',
    None,
    'Number of qubit rows in random circuit to benchmark.',
    lower_bound=2)

absl_flags.DEFINE_integer(
    'n_cols',
    None,
    'Number of qubit columns in random circuit to benchmark.',
    lower_bound=2)

absl_flags.DEFINE_integer('batch_size',
                          1,
                          'The number of circuits to simulate in parallel.',
                          lower_bound=1)

absl_flags.DEFINE_integer(
    'n_iters',
    1, "Number of rounds to run each benchmark, corresponding to"
    "number of iterations in a training context. ",
    lower_bound=1)

# Benchmark metadata.
absl_flags.DEFINE_string('backend', None,
                         'Which backend simulator to benchmark.')

absl_flags.DEFINE_integer(
    'n_runs',
    1,
    'Number of times to run the model for its specified number of iterations '
    'during benchmarking. For example, if a model is specified to be trained '
    'for 50 iterations, `n_runs=10` would reset this model after training a '
    'total of 10 times, resulting in a time overhead of 500 total iterations.',
    lower_bound=1)

absl_flags.DEFINE_integer('n_burn',
                          0,
                          'Number of burner runs. See `n_runs`.',
                          lower_bound=0)


def test_flags(**kwargs):
    """Create a set of test flags by kwarg assignment.

    This constructs a named tuple that mimics the interface of absl.flags.
    Any command line flags defined with defaults will be present in the output
    with their default value unless overwritten.

    Returns:
        namedtuple containing valid flag names.
    """
    base_flags = FLAGS.flag_values_dict()
    updated = dict(base_flags, **kwargs)
    return namedtuple('params', updated.keys())(**updated)
