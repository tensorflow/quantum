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
        for nspins in supported_nspins_tfi_chain:
            circuits, _, _, addinfo = data_dict_tfi_chain[nspins]
            for n in random_subset_tfi_chain:
                phi = cirq.Simulator().simulate(circuits[n]).final_state
                gs = addinfo[n].gs
                self.assertAllClose(np.abs(np.vdot(gs, phi)), 1.0, rtol=1e-3)

    def test_paulisum(self):
        """Test that the PauliSum Hamiltonians give the ground state energy."""
        for nspins in supported_nspins_tfi_chain:
            circuits, _, pauli_sums, addinfo = data_dict_tfi_chain[nspins]
            qubit_map = {
                qbs_dict_tfi_chain[nspins][i]: i for i in range(nspins)
            }
            for n in random_subset_tfi_chain:
                phi = cirq.Simulator().simulate(circuits[n]).final_state
                e = pauli_sums[n].expectation_from_wavefunction(phi, qubit_map)
                self.assertAllClose(e, addinfo[n].gs_energy, rtol=1e-4)

    def test_returned_objects(self):
        """Test that the length and types of returned objects are correct."""
        for nspins in supported_nspins_tfi_chain:
            circuits, labels, pauli_sums, addinfo = data_dict_tfi_chain[nspins]
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
        for nspins in supported_nspins_tfi_chain:
            circuits, _, _, addinfo = data_dict_tfi_chain[nspins]
            for n in random_subset_tfi_chain:
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
        for nspins in supported_nspins_xxz_chain:
            circuits, _, _, addinfo = data_dict_xxz_chain[nspins]
            for n in random_subset_xxz_chain:
                phi = cirq.Simulator().simulate(circuits[n]).final_state
                gs = addinfo[n].gs
                self.assertAllClose(np.abs(np.vdot(gs, phi)), 1.0, rtol=5e-3)

    def test_paulisum(self):
        """Test that the PauliSum Hamiltonians give the ground state energy."""
        for nspins in supported_nspins_xxz_chain:
            circuits, _, pauli_sums, addinfo = data_dict_xxz_chain[nspins]
            qubit_map = {
                qbs_dict_xxz_chain[nspins][i]: i for i in range(nspins)
            }
            for n in random_subset_xxz_chain:
                phi = cirq.Simulator().simulate(circuits[n]).final_state
                e = pauli_sums[n].expectation_from_wavefunction(phi, qubit_map)
                self.assertAllClose(e, addinfo[n].gs_energy, rtol=5e-3)

    def test_returned_objects(self):
        """Test that the length and types of returned objects are correct."""
        for nspins in supported_nspins_xxz_chain:
            circuits, labels, pauli_sums, addinfo = data_dict_xxz_chain[nspins]
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
        for nspins in supported_nspins_xxz_chain:
            circuits, _, _, addinfo = data_dict_xxz_chain[nspins]
            for n in random_subset_xxz_chain:
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
    random_subset_size = 10

    # TFI CHAIN
    supported_nspins_tfi_chain = [4, 8, 12, 16]
    data_dict_tfi_chain = {}
    qbs_dict_tfi_chain = {}
    for nspins in supported_nspins_tfi_chain:
        qbs_tfi_chain = cirq.GridQubit.rect(nspins, 1)
        data_dict_tfi_chain[nspins] = spin_system.tfi_chain(
            qbs_tfi_chain,
            'closed',
        )
        qbs_dict_tfi_chain[nspins] = qbs_tfi_chain
    random_subset_tfi_chain = np.random.permutation(list(
        range(81)))[:random_subset_size]

    # XXZ CHAIN
    supported_nspins_xxz_chain = [4, 8, 12, 16]
    data_dict_xxz_chain = {}
    qbs_dict_xxz_chain = {}
    for nspins in supported_nspins_xxz_chain:
        qbs_xxz_chain = cirq.GridQubit.rect(nspins, 1)
        data_dict_xxz_chain[nspins] = spin_system.xxz_chain(
            qbs_xxz_chain,
            'closed',
        )
        qbs_dict_xxz_chain[nspins] = qbs_xxz_chain
    random_subset_xxz_chain = np.random.permutation(list(
        range(76)))[:random_subset_size]

    tf.test.main()
