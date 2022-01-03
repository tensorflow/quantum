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
# Remove PYTHONPATH collisions for protobuf.
import sys
new_path = [x for x in sys.path if 'com_google_protobuf' not in x]
sys.path = new_path

import tensorflow as tf
import numpy as np
import cirq
from tensorflow_quantum.datasets import spin_system
from tensorflow_quantum.datasets.spin_system import SpinSystemInfo


class TFIChainTest(tf.test.TestCase):
    """Testing tfi_chain."""
    # pylint: disable=C0103

    @classmethod
    def setUpClass(self):
        """Setup data for the test"""
        self.random_subset_size = 10
        self.supported_nspins_tfi_chain = [4, 8, 12, 16]
        self.data_dict_tfi_chain = {}
        self.qbs_dict_tfi_chain = {}
        for nspins in self.supported_nspins_tfi_chain:
            qbs_tfi_chain = cirq.GridQubit.rect(nspins, 1)
            self.data_dict_tfi_chain[nspins] = spin_system.tfi_chain(
                qbs_tfi_chain,
                'closed',
            )
            self.qbs_dict_tfi_chain[nspins] = qbs_tfi_chain
        self.random_subset_tfi_chain = np.random.permutation(list(
            range(76)))[:self.random_subset_size]
        super(TFIChainTest).__init__(self)

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
        for nspins in self.supported_nspins_tfi_chain:
            circuits, _, _, addinfo = self.data_dict_tfi_chain[nspins]
            for n in self.random_subset_tfi_chain:
                phi = cirq.Simulator().simulate(circuits[n]).final_state_vector
                gs = addinfo[n].gs
                self.assertAllClose(np.abs(np.vdot(gs, phi)), 1.0, rtol=1e-3)

    def test_paulisum(self):
        """Test that the PauliSum Hamiltonians give the ground state energy."""
        for nspins in self.supported_nspins_tfi_chain:
            circuits, _, pauli_sums, addinfo = self.data_dict_tfi_chain[nspins]
            qubit_map = {
                self.qbs_dict_tfi_chain[nspins][i]: i for i in range(nspins)
            }
            for n in self.random_subset_tfi_chain:
                phi = cirq.Simulator().simulate(circuits[n]).final_state_vector
                e = pauli_sums[n].expectation_from_state_vector(phi, qubit_map)
                self.assertAllClose(e, addinfo[n].gs_energy, rtol=1e-4)

    def test_returned_objects(self):
        """Test that the length and types of returned objects are correct."""
        for nspins in self.supported_nspins_tfi_chain:
            circuits, labels, pauli_sums, addinfo = self.data_dict_tfi_chain[
                nspins]
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
        for nspins in self.supported_nspins_tfi_chain:
            circuits, _, _, addinfo = self.data_dict_tfi_chain[nspins]
            for n in self.random_subset_tfi_chain:
                resolved_circuit = cirq.resolve_parameters(
                    addinfo[n].var_circuit, addinfo[n].params)
                state_circuit = cirq.Simulator().simulate(
                    circuits[n]).final_state_vector
                state_resolved_circuit = cirq.Simulator().simulate(
                    resolved_circuit).final_state_vector
                self.assertAllClose(np.abs(
                    np.vdot(state_circuit, state_resolved_circuit)),
                                    1.0,
                                    rtol=1e-3)


class XXZChainTest(tf.test.TestCase):
    """Testing xxz_chain."""
    # pylint: disable=C0103

    @classmethod
    def setUpClass(self):
        """Setup data for the test"""
        self.random_subset_size = 10
        # XXZ CHAIN
        self.supported_nspins_xxz_chain = [4, 8, 12, 16]
        self.data_dict_xxz_chain = {}
        self.qbs_dict_xxz_chain = {}
        for nspins in self.supported_nspins_xxz_chain:
            qbs_xxz_chain = cirq.GridQubit.rect(nspins, 1)
            self.data_dict_xxz_chain[nspins] = spin_system.xxz_chain(
                qbs_xxz_chain,
                'closed',
            )
            self.qbs_dict_xxz_chain[nspins] = qbs_xxz_chain
        self.random_subset_xxz_chain = np.random.permutation(list(
            range(76)))[:self.random_subset_size]
        super(XXZChainTest).__init__(self)

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
        for nspins in self.supported_nspins_xxz_chain:
            circuits, _, _, addinfo = self.data_dict_xxz_chain[nspins]
            for n in self.random_subset_xxz_chain:
                phi = cirq.Simulator().simulate(circuits[n]).final_state_vector
                gs = addinfo[n].gs
                self.assertAllClose(np.abs(np.vdot(gs, phi)), 1.0, rtol=5e-3)

    def test_paulisum(self):
        """Test that the PauliSum Hamiltonians give the ground state energy."""
        for nspins in self.supported_nspins_xxz_chain:
            circuits, _, pauli_sums, addinfo = self.data_dict_xxz_chain[nspins]
            qubit_map = {
                self.qbs_dict_xxz_chain[nspins][i]: i for i in range(nspins)
            }
            for n in self.random_subset_xxz_chain:
                phi = cirq.Simulator().simulate(circuits[n]).final_state_vector
                e = pauli_sums[n].expectation_from_state_vector(phi, qubit_map)
                self.assertAllClose(e, addinfo[n].gs_energy, rtol=5e-3)

    def test_returned_objects(self):
        """Test that the length and types of returned objects are correct."""
        for nspins in self.supported_nspins_xxz_chain:
            circuits, labels, pauli_sums, addinfo = self.data_dict_xxz_chain[
                nspins]
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
        for nspins in self.supported_nspins_xxz_chain:
            circuits, _, _, addinfo = self.data_dict_xxz_chain[nspins]
            for n in self.random_subset_xxz_chain:
                resolved_circuit = cirq.resolve_parameters(
                    addinfo[n].var_circuit, addinfo[n].params)
                state_circuit = cirq.Simulator().simulate(
                    circuits[n]).final_state_vector
                state_resolved_circuit = cirq.Simulator().simulate(
                    resolved_circuit).final_state_vector
                self.assertAllClose(np.abs(
                    np.vdot(state_circuit, state_resolved_circuit)),
                                    1.0,
                                    rtol=1e-3)


class TFIRectangularTest(tf.test.TestCase):
    """Testing tfi_rectangular."""
    # pylint: disable=C0103

    @classmethod
    def setUpClass(self):
        """Setup data for the test"""
        self.random_subset_size = 10
        # TFI RECT
        self.supported_nspins_tfi_rectangular = [9, 12, 16]
        self.data_dict_tfi_rectangular = {}
        self.qbs_dict_tfi_rectangular = {}
        for nspins in self.supported_nspins_tfi_rectangular:
            qbs_tfi_rectangular = cirq.GridQubit.rect(nspins, 1)
            self.data_dict_tfi_rectangular[
                nspins] = spin_system.tfi_rectangular(
                    qbs_tfi_rectangular,
                    'torus',
                )
            self.qbs_dict_tfi_rectangular[nspins] = qbs_tfi_rectangular
        self.random_subset_tfi_rectangular = np.random.permutation(
            list(range(51)))[:self.random_subset_size]
        super(TFIRectangularTest).__init__(self)

    def test_errors(self):
        """Test that it errors on invalid arguments."""
        with self.assertRaisesRegex(ValueError,
                                    expected_regex='Supported number of'):
            qbs = cirq.GridQubit.rect(3, 1)
            spin_system.tfi_rectangular(qbs)
        with self.assertRaisesRegex(
                ValueError, expected_regex='Supported boundary conditions'):
            qbs = cirq.GridQubit.rect(9, 1)
            spin_system.tfi_rectangular(qbs, 'open')
        with self.assertRaisesRegex(TypeError,
                                    expected_regex='expected str, bytes '):
            qbs = cirq.GridQubit.rect(9, 1)
            spin_system.tfi_rectangular(qbs, data_dir=123)
        with self.assertRaisesRegex(TypeError,
                                    expected_regex='must be a list of'):
            spin_system.tfi_rectangular(['bob'])

        with self.assertRaisesRegex(TypeError,
                                    expected_regex='must be a one-dimensional'):
            qbs = cirq.GridQubit.rect(9, 1)
            spin_system.tfi_rectangular([qbs])

    def test_fidelity(self):
        """Test that all fidelities are close to 1."""
        for nspins in self.supported_nspins_tfi_rectangular:
            circuits, _, _, addinfo = self.data_dict_tfi_rectangular[nspins]
            for n in self.random_subset_tfi_rectangular:
                phi = cirq.Simulator().simulate(circuits[n]).final_state_vector
                gs = addinfo[n].gs
                self.assertAllClose(np.abs(np.vdot(gs, phi)), 1.0, rtol=5e-3)

    def test_paulisum(self):
        """Test that the PauliSum Hamiltonians give the ground state energy."""
        for nspins in self.supported_nspins_tfi_rectangular:
            circuits, _, pauli_sums, addinfo = self.data_dict_tfi_rectangular[
                nspins]
            qubit_map = {
                self.qbs_dict_tfi_rectangular[nspins][i]: i
                for i in range(nspins)
            }
            for n in self.random_subset_tfi_rectangular:
                phi = cirq.Simulator().simulate(circuits[n]).final_state_vector
                e = pauli_sums[n].expectation_from_state_vector(phi, qubit_map)
                self.assertAllClose(e, addinfo[n].gs_energy, rtol=5e-3)

    def test_returned_objects(self):
        """Test that the length and types of returned objects are correct."""
        for nspins in self.supported_nspins_tfi_rectangular:
            circuits, labels, pauli_sums, addinfo = \
                self.data_dict_tfi_rectangular[nspins]
            self.assertLen(circuits, 51)
            self.assertLen(labels, 51)
            self.assertLen(pauli_sums, 51)
            self.assertLen(addinfo, 51)
            for n in range(51):
                self.assertIsInstance(circuits[n], cirq.Circuit)
                self.assertIsInstance(labels[n], int)
                self.assertIsInstance(pauli_sums[n], cirq.PauliSum)
                self.assertIsInstance(addinfo[n], SpinSystemInfo)

    def test_param_resolver(self):
        """Test that the resolved circuits are correct."""
        for nspins in self.supported_nspins_tfi_rectangular:
            circuits, _, _, addinfo = self.data_dict_tfi_rectangular[nspins]
            for n in self.random_subset_tfi_rectangular:
                resolved_circuit = cirq.resolve_parameters(
                    addinfo[n].var_circuit, addinfo[n].params)
                state_circuit = cirq.Simulator().simulate(
                    circuits[n]).final_state_vector
                state_resolved_circuit = cirq.Simulator().simulate(
                    resolved_circuit).final_state_vector
                self.assertAllClose(np.abs(
                    np.vdot(state_circuit, state_resolved_circuit)),
                                    1.0,
                                    rtol=1e-3)


if __name__ == '__main__':

    tf.test.main()
