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
import tensorflow as tf
import numpy as np
import cirq
from tensorflow_quantum.datasets import spin_system
from tensorflow_quantum.datasets.spin_system import SpinSystemInfo
import logging
import time

class TFIChainTest(tf.test.TestCase):
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
        for nspins in SUPPORTED_NSPINS_TFI_CHAIN:
            circuits, _, _, addinfo = DATA_DICT_TFI_CHAIN[nspins]
            for n in range(len(addinfo)):
                phi = cirq.Simulator().simulate(circuits[n]).final_state
                gs = addinfo[n].gs
                self.assertAllClose(np.abs(np.vdot(gs, phi)), 1.0, rtol=1e-3)

    def test_paulisum(self):
        """Test that the PauliSum Hamiltonians give the ground state energy."""
        for nspins in SUPPORTED_NSPINS_TFI_CHAIN:
            circuits, _, pauli_sums, addinfo = DATA_DICT_TFI_CHAIN[nspins]
            qubit_map = {
                QBS_DICT_TFI_CHAIN[nspins][i]: i for i in range(nspins)
            }
            for n in range(len(addinfo)):
                phi = cirq.Simulator().simulate(circuits[n]).final_state
                e = pauli_sums[n].expectation_from_wavefunction(phi, qubit_map)
                self.assertAllClose(e, addinfo[n].gs_energy, rtol=1e-4)

    def test_returned_objects(self):
        """Test that the length and types of returned objects are correct."""
        for nspins in SUPPORTED_NSPINS_TFI_CHAIN:
            circuits, labels, pauli_sums, addinfo = DATA_DICT_TFI_CHAIN[nspins]
            self.assertLen(circuits, 81)
            self.assertLen(labels, 81)
            self.assertLen(pauli_sums, 81)
            self.assertLen(addinfo, 81)
            for n in range(81):
                self.assertIsInstance(circuits[n], cirq.Circuit)
                self.assertIsInstance(labels[n], int)
                self.assertIsInstance(pauli_sums[n], cirq.PauliSum)
                self.assertIsInstance(addinfo[n], SpinSystemInfo)

    def test_param_resolver(self):
        """Test that the resolved circuits are correct."""
        for nspins in SUPPORTED_NSPINS_TFI_CHAIN:
            circuits, _, _, addinfo = DATA_DICT_TFI_CHAIN[nspins]
            for n in range(len(addinfo)):
                resolved_circuit = cirq.resolve_parameters(
                    addinfo[n].var_circuit, addinfo[n].params)
                state_circuit = cirq.Simulator().simulate(
                    circuits[n]).final_state
                state_resolved_circuit = cirq.Simulator().simulate(
                    resolved_circuit).final_state
                self.assertAllClose(np.abs(
                    np.vdot(state_circuit, state_resolved_circuit)),
                                    1.0,
                                    rtol=1e-3)


class XXZChainTest(tf.test.TestCase):
    """Testing xxz_chain."""

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
        for nspins in SUPPORTED_NSPINS_XXZ_CHAIN:
            circuits, _, _, addinfo = DATA_DICT_XXZ_CHAIN[nspins]
            for n in range(len(addinfo)):
                phi = cirq.Simulator().simulate(circuits[n]).final_state
                gs = addinfo[n].gs
                self.assertAllClose(np.abs(np.vdot(gs, phi)), 1.0, rtol=5e-3)

    def test_paulisum(self):
        """Test that the PauliSum Hamiltonians give the ground state energy."""
        for nspins in SUPPORTED_NSPINS_XXZ_CHAIN:
            circuits, _, pauli_sums, addinfo = DATA_DICT_XXZ_CHAIN[nspins]
            qubit_map = {
                QBS_DICT_XXZ_CHAIN[nspins][i]: i for i in range(nspins)
            }
            for n in range(len(addinfo)):
                phi = cirq.Simulator().simulate(circuits[n]).final_state
                e = pauli_sums[n].expectation_from_wavefunction(phi, qubit_map)
                self.assertAllClose(e, addinfo[n].gs_energy, rtol=5e-3)

    def test_returned_objects(self):
        """Test that the length and types of returned objects are correct."""
        for nspins in SUPPORTED_NSPINS_XXZ_CHAIN:
            circuits, labels, pauli_sums, addinfo = DATA_DICT_XXZ_CHAIN[nspins]
            self.assertLen(circuits, 76)
            self.assertLen(labels, 76)
            self.assertLen(pauli_sums, 76)
            self.assertLen(addinfo, 76)
            for n in range(76):
                self.assertIsInstance(circuits[n], cirq.Circuit)
                self.assertIsInstance(labels[n], int)
                self.assertIsInstance(pauli_sums[n], cirq.PauliSum)
                self.assertIsInstance(addinfo[n], SpinSystemInfo)

    def test_param_resolver(self):
        """Test that the resolved circuits are correct."""
        for nspins in SUPPORTED_NSPINS_XXZ_CHAIN:
            circuits, _, _, addinfo = DATA_DICT_XXZ_CHAIN[nspins]
            for n in range(len(addinfo)):
                resolved_circuit = cirq.resolve_parameters(
                    addinfo[n].var_circuit, addinfo[n].params)
                state_circuit = cirq.Simulator().simulate(
                    circuits[n]).final_state
                state_resolved_circuit = cirq.Simulator().simulate(
                    resolved_circuit).final_state
                self.assertAllClose(np.abs(
                    np.vdot(state_circuit, state_resolved_circuit)),
                                    1.0,
                                    rtol=1e-3)


if __name__ == '__main__':

    # TFI CHAIN
    SUPPORTED_NSPINS_TFI_CHAIN = [4, 8, 12, 16]
    DATA_DICT_TFI_CHAIN = {}
    QBS_DICT_TFI_CHAIN = {}
    for nspins in SUPPORTED_NSPINS_TFI_CHAIN:
        QBS_TFI_CHAIN = cirq.GridQubit.rect(nspins, 1)
        logging.warning('\nCalling tfi_chain, nspins={}\n'.format(nspins))
        start = time.time()
        DATA_DICT_TFI_CHAIN[nspins] = spin_system.tfi_chain(
            QBS_TFI_CHAIN,
            'closed',
        )
        logging.warning('\nEnd of function call, duration {}\n'.format(time.time()-start))
        QBS_DICT_TFI_CHAIN[nspins] = QBS_TFI_CHAIN


    # XXZ CHAIN
    SUPPORTED_NSPINS_XXZ_CHAIN = [4, 8, 12, 16]
    DATA_DICT_XXZ_CHAIN = {}
    QBS_DICT_XXZ_CHAIN = {}
    for nspins in SUPPORTED_NSPINS_XXZ_CHAIN:
        QBS_XXZ_CHAIN = cirq.GridQubit.rect(nspins, 1)
        logging.warning('\nCalling xxz_chain, nspins={}\n'.format(nspins))
        start = time.time()
        DATA_DICT_XXZ_CHAIN[nspins] = spin_system.xxz_chain(
            QBS_XXZ_CHAIN,
            'closed',
        )
        logging.warning('\nEnd of function call, duration {}\n'.format(time.time()-start))

        QBS_DICT_XXZ_CHAIN[nspins] = QBS_XXZ_CHAIN

    tf.test.main()
