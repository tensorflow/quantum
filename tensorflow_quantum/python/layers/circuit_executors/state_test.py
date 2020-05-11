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
"""Tests for tensorflow_quantum.layers.circuit_executors.state."""
import numpy as np
from absl.testing import parameterized
import sympy
import tensorflow as tf
import cirq

from tensorflow_quantum.python.layers.circuit_executors import state
from tensorflow_quantum.python import util

WF_OUTPUT = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
DM_OUTPUT = np.outer(WF_OUTPUT, WF_OUTPUT)


class StateTest(parameterized.TestCase, tf.test.TestCase):
    """Basic tests for the State layer."""

    def test_state_create(self):
        """Test that State layers can be created."""
        state.State()
        state.State(backend=cirq.Simulator())
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="junk is invalid"):
            state.State('junk')

    @parameterized.parameters([{
        'backend': None
    }, {
        'backend': cirq.Simulator()
    }, {
        'backend': cirq.DensityMatrixSimulator()
    }])
    def test_state_invalid_combinations(self, backend):
        """Test with valid type inputs and valid value, but incorrect combo."""
        state_calc = state.State(backend)
        symbol = sympy.Symbol('alpha')
        circuit = cirq.Circuit(cirq.H(cirq.GridQubit(0, 0))**symbol)
        with self.assertRaisesRegex(Exception, expected_regex=""):
            # no value provided.
            state_calc([circuit, circuit], symbol_names=[symbol], repetitions=5)

        with self.assertRaisesRegex(Exception, expected_regex=""):
            # no name provided.
            state_calc([circuit, circuit],
                       symbol_names=[],
                       symbol_values=[[2.0], [3.0]])

        with self.assertRaisesRegex(Exception, expected_regex=""):
            # deceptive, but the circuit shouldn't be in a list. otherwise fine.
            state_calc([circuit],
                       symbol_names=['alpha'],
                       symbol_values=[[2.0], [3.0]])

        with self.assertRaisesRegex(Exception, expected_regex=""):
            # wrong symbol name.
            state_calc([circuit],
                       symbol_names=['alphaaaa'],
                       symbol_values=[[2.0], [3.0]])

        with self.assertRaisesRegex(Exception, expected_regex=""):
            # too many symbol values provided.
            state_calc(circuit,
                       symbol_names=['alpha'],
                       symbol_values=[[2.0, 4.0], [3.0, 5.0]])

    def test_state_basic_inputs(self):
        """Test that state ingests inputs correctly in simple settings."""
        state_calc = state.State()
        state_calc(cirq.Circuit())
        state_calc([cirq.Circuit()])
        state_calc(cirq.Circuit(), symbol_names=['name'], symbol_values=[[0.5]])
        state_calc(cirq.Circuit(),
                   symbol_names=[sympy.Symbol('name')],
                   symbol_values=[[0.5]])

    def test_sample_outputs_simple(self):
        """Test the simplest call where nothing but circuits are provided."""
        state_calc = state.State()
        circuit = cirq.Circuit(cirq.H(cirq.GridQubit(0, 0)))
        output = state_calc([circuit, circuit])
        self.assertShapeEqual(np.empty((2, 2)), output.to_tensor())

    @parameterized.parameters([
        {
            'backend_output': (None, WF_OUTPUT)
        },
        {
            'backend_output': (cirq.sim.sparse_simulator.Simulator(), WF_OUTPUT)
        },
        {
            'backend_output':
                (cirq.sim.density_matrix_simulator.DensityMatrixSimulator(),
                 DM_OUTPUT)
        },
    ])
    def test_state_output(self, backend_output):
        """Check that any output type is as expected.

        This layer only allows for 2 different outputs, depending on whether a
        wavefuntion or density matrix simulator is used. Therefore any pre or
        post processing done inside the layers should not cause output from the
        layer to structurally deviate from what is expected.
        """
        backend = backend_output[0]
        output = backend_output[1]
        state_executor = state.State(backend=backend)
        bits = cirq.GridQubit.rect(1, 2)
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on(bits[0]))
        circuit.append(cirq.CNOT(bits[0], bits[1]))
        programs = util.convert_to_tensor([circuit, circuit])
        layer_output = state_executor(programs).to_list()
        self.assertAllClose(layer_output, [output, output])

    def test_state_one_circuit(self):
        """Test that State behaves when a single layer is specified."""
        state_calc = state.State()
        state_calc(cirq.Circuit(),
                   symbol_values=tf.zeros((5, 0), dtype=tf.dtypes.float32))


if __name__ == '__main__':
    tf.test.main()
