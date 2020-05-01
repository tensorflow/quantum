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
"""Quantum datasets for several quantum many-body spin systems in the form of
parameterized circuits"""
import os
import numpy as np
import sympy
import scipy.sparse as sps
import cirq


class SpinSystem(object):
    """Abstract spin system class containing information about the quantum
    data
    """

    def __init__(self, g: float, res_e: float, fidelity: float,
                 gs_energy: float, parameters: np.ndarray, path: str):
        """Instantiate this spin system.

        Args:
            g: Order parameter
            res_e: Residual energy between the exact diagonalization ground
            state and the circuit state.
            fidelity: Fidelity between the exact diagonalization ground state
            and the circuit state.
            gs_energy: Exact diagonalization  round state energy
            parameters: Numpy array of shape (M,depth) where M is the required
             number of parameters per layer.
            path: Path to the data.
        """
        self.g = g
        self.res_e = res_e
        self.fidelity = fidelity
        self.gs_energy = gs_energy
        self.parameters = parameters
        self.path = path

    def __lt__(self, other):
        return self.g < other.g

    def hamiltonian(self):
        """Load the hamiltonian into a scipy sparse CSR matrix."""
        return sps.load_npz(self.path + '/H.npz')

    def ground_state(self):
        """Load the ground state wave function into a complex numpy array."""
        return np.load(self.path + '/groundstate.npy')


def unique_name():
    """
    Generator to generate an infinite number of unique names.

    Yields:
        Strings of the form 'theta_<integer>'

    """
    num = 0
    while True:
        yield 'theta_' + str(num)
        num += 1


class SpinSystemDataset(object):
    """Abstract class that handles a single instance of the system for a
    single set of order parameters.
    """

    def __init__(self, nspins: int, system_name: str):
        """Instantiate this data set.

        Args:
            nspins: The number of spins in the system.
            system_name: Name of the system we want to load.
        """

        # TODO: How should downloading and storing the data be handled?
        data_path = '/tmp/data/' + system_name + "_{}/".format(int(nspins))
        self.nspins = nspins
        if not os.path.exists(data_path):
            os.mkdir(data_path)
            self._download_dataset(data_path)
        # TODO: This only works for a single order parameter. For
        #  the XY model or XYZ model, this will have to be generalized.
        self.systems = self._get_spin_systems(data_path)

    def __str__(self):
        # TODO: This only works for a single order parameter. For
        #  the XY model or XYZ model, this will have to be generalized.
        return "System with {} order parameters in range [{},{}]".format(
            len(self), self.systems[0].g, self.systems[-1].g)

    def __len__(self):
        return len(self.systems)

    def _get_spin_systems(self, path: str):
        systems = []
        for directory in os.listdir(path):
            with open(path + directory + '/stats.txt', 'r') as file:
                g = float(directory)
                lines = file.readlines()
                res_e = float(lines[0].split('=')[1].strip('\n'))
                fidelity = float(lines[0].split('=')[1].strip('\n'))
            gs_energy = np.load(path + directory + '/energy.npy')[0]
            parameters = np.load(path + directory + '/params.npy')
            systems.append(
                SpinSystem(g, res_e, fidelity, gs_energy, parameters,
                           path + directory))
        return sorted(systems)

    def _download_dataset(self, path: str):
        # TODO: Implement this.
        pass


def tfi_chain(qubits, nspins: int, boundary_condition: str = 'closed'):
    """
    Transverse field Ising-model quantum data set

    H = - \sum_{i} \sigma_i^z \sigma_{i+1}^z - g \sum_i \sigma_i^x

    Contains 80 circuit parameterizations corresponding to
    the ground states of the 1D TFI chain for g in [0.2,1.8].

    Args:
        qubits: cirq.Grid qubits of size (nspins,1)
        nspins: Number of spins, supported numbers are [4,8,12,16]
        boundary_condition: The boundary condition of the chain. Supported
        boundary conditions are ['closed']

    Returns:
        cirq.Circuit of depth N/2.
        List of resolved parameters for all 80 order parameters.
        List of SpinSystem instances.
    """

    supported_n = [4, 8, 12, 16]
    supported_bc = ['closed']
    if not isinstance(nspins, int):
        raise TypeError("nspins must be an integer, received {}".format(
            type(nspins)))
    if not nspins in supported_n:
        raise ValueError('Supported number of spins are {}, received {}'.format(
            supported_n, nspins))

    if not boundary_condition in supported_bc:
        raise ValueError(
            'Supported boundary conditions are {}, received {}'.format(
                supported_bc, boundary_condition))

    boundary_condition = boundary_condition
    name = 'TFI_' + boundary_condition

    dataset = SpinSystemDataset(nspins, name)

    name_generator = unique_name()
    symbols = [sympy.Symbol(next(name_generator)) for _ in range(nspins)]
    symbol_names = np.array([s.name for s in symbols]).reshape(
        (2, int(nspins / 2)))
    symbols = np.array(symbols).reshape((2, int(nspins / 2)))
    resolved_parameters = []
    for i in range(len(dataset.systems)):
        parameter_map = dict(
            zip(symbol_names.flatten(),
                dataset.systems[i].parameters.flatten()))
        resolved_parameters.append(cirq.ParamResolver(parameter_map))

    # define the circuit
    if not all([isinstance(q, cirq.GridQubit) for q in qubits]):
        raise TypeError("qubits must be a list of cirq.Gridqubit objects.")
    if not all([q.col == 0 for q in qubits]):
        raise ValueError(
            'Must be list of cirq.Gridqubit objects with shape (i,0)')
    if not len(qubits) == nspins:
        raise ValueError(
            "Expected {} cirq.Gridqubit objects, received {}".format(
                nspins, len(qubits)))
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(qubits))

    for d in range(int(nspins / 2)):
        for j in range(0, nspins - 1, 2):
            circuit.append(cirq.CNOT(qubits[j], qubits[j + 1]))
            circuit.append(cirq.rz(symbols[0, d])(qubits[j + 1]))
            circuit.append(cirq.CNOT(qubits[j], qubits[j + 1]))
        for j in range(1, nspins - 1, 2):
            circuit.append(cirq.CNOT(qubits[j], qubits[j + 1]))
            circuit.append(cirq.rz(symbols[0, d])(qubits[j + 1]))
            circuit.append(cirq.CNOT(qubits[j], qubits[j + 1]))
        if boundary_condition == 'closed':
            circuit.append(cirq.CNOT(qubits[nspins - 1], qubits[0]))
            circuit.append(cirq.rz(symbols[0, d])(qubits[0]))
            circuit.append(cirq.CNOT(qubits[nspins - 1], qubits[0]))
        for j in range(0, nspins):
            circuit.append(cirq.rx(symbols[1, d])(qubits[j]))

    return circuit, resolved_parameters, dataset.systems
