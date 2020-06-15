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
"""Test the spin system dataset"""
from collections import namedtuple
import tensorflow as tf
import numpy as np
import cirq
from tensorflow_quantum.datasets import spin_system
from tensorflow_quantum.datasets.spin_system import SpinSystemInfo


class TFI_ChainTest(tf.test.TestCase):
    """Testing tfi_chain."""

    def test_errors(self):
        """Test that it errors on invalid arguments."""
        with self.assertRaisesRegex(ValueError,
                                    expected_regex='Supported number of'):
            qbs = cirq.GridQubit.rect(3, 1)
            spin_system.tfi_chain(qbs)
        with self.assertRaisesRegex(
                ValueError, expected_regex='Supported boundary conditions'):
            qbs = cirq.GridQubit.rect(4, 1)
            spin_system.tfi_chain(qbs, 'open')
        with self.assertRaisesRegex(TypeError,
                                    expected_regex='expected str, bytes '):
            qbs = cirq.GridQubit.rect(4, 1)
            spin_system.tfi_chain(qbs, data_dir=123)
        with self.assertRaisesRegex(TypeError,
                                    expected_regex='must be a list of'):
            spin_system.tfi_chain(['bob'])

        with self.assertRaisesRegex(TypeError,
                                    expected_regex='must be a one-dimensional'):
            qbs = cirq.GridQubit.rect(4, 1)
            spin_system.tfi_chain([qbs])

    def test_fidelity(self):
        """Test that all fidelities are close to 1."""
        supported_nspins = [4, 8, 12, 16]
        for nspins in supported_nspins:
            qbs = cirq.GridQubit.rect(nspins, 1)
            circuits, _, _, addinfo = spin_system.tfi_chain(
                qbs,
                'closed',
            )
            for n in range(len(addinfo)):
                phi = cirq.Simulator().simulate(circuits[n]).final_state
                gs = addinfo[n].gs
                self.assertAllClose(np.abs(np.conj(np.dot(gs, phi))),
                                    1.0,
                                    rtol=1e-3)

    def test_paulisum(self):
        """Test that the PauliSum Hamiltonians give the ground state energy."""
        supported_nspins = [4, 8, 12, 16]
        for nspins in supported_nspins:
            qbs = cirq.GridQubit.rect(nspins, 1)
            circuits, _, pauli_sums, addinfo = spin_system.tfi_chain(
                qbs, 'closed')
            qubit_map = {qbs[i]: i for i in range(len(qbs))}
            for n in range(len(pauli_sums)):
                phi = cirq.Simulator().simulate(circuits[n]).final_state
                e = pauli_sums[n].expectation_from_wavefunction(phi, qubit_map)
                self.assertAllClose(e, addinfo[n].gs_energy, rtol=1e-4)

    def test_returned_objects(self):
        """Test that the length and types of returned objects are correct."""
        supported_nspins = [4, 8, 12, 16]
        for nspins in supported_nspins:
            qbs = cirq.GridQubit.rect(nspins, 1)
            circuits, labels, pauli_sums, addinfo = spin_system.tfi_chain(
                qbs, 'closed')
            self.assertLen(circuits, 81)
            self.assertLen(labels, 81)
            self.assertLen(pauli_sums, 81)
            self.assertLen(addinfo, 81)
            for n in range(81):
                self.assertIsInstance(circuits[n], cirq.Circuit)
                self.assertIsInstance(labels[n], int)
                self.assertIsInstance(pauli_sums[n], cirq.PauliSum)
                self.assertIsInstance(addinfo[n], SpinSystemInfo)


if __name__ == '__main__':
    tf.test.main()
