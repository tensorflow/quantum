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
"""Quantum datasets for quantum many-body spin systems"""
from pathlib import Path
import numpy as np
import sympy
import cirq


class SpinSystem(object):
    """Abstract spin system class."""

    def __init__(self, g, gs_energy, parameters, path, hamiltonian_fn):
        """Instantiate this spin system.

        This object contains all the information about a single instance of
        a quantum many body system.

        Args:
            g: Python `float` order parameter
            gs_energy: Python `float` Ground state energy from exact
                diagonalization.
            parameters: Numpy array of shape `(M, depth)` where M is the
                required number of parameters per layer.
            path: Path to the data.
            hamiltonian_fn: Python `lambda` function that accepts an order
                parameter and returns the corresponding `PauliSum` for the
                Hamiltonian.
        """
        self.g = g
        self.gs_energy = gs_energy
        self.parameters = parameters
        self.path = path
        self._hamiltonian_fn = hamiltonian_fn

    def __lt__(self, other):
        return self.g < other.g

    def hamiltonian(self, qubits):
        """Get a cirq.Paulisum object that represents the Hamiltonian"""
        return self._hamiltonian_fn(qubits, self.g)

    def ground_state(self):
        """Load the ground state wave function into a complex numpy array."""
        return np.load(self.path / Path('groundstate.npy'))


def unique_name():
    """Generator to generate an infinite number of unique names.

    Yields:
        Python `str` of the form 'theta_<integer>'

    """
    num = 0
    while True:
        yield 'theta_' + str(num)
        num += 1


class SpinSystemDataset(object):
    """Abstract class that handles a single instance of many-body system"""

    def __init__(self, nspins, system_name, data_dir, hamiltonian_fn):
        """Instantiate this data set.

        Args:
            nspins: The number of spins in the system.
            system_name: Name of the system we want to load.
            data_dir: Location where to store the data on disk. Default is
                ~/tfq-datasets
        """

        # TODO: How should downloading and storing the data be handled?
        if data_dir is None:
            data_dir = Path.expanduser(Path('~/tfq-datasets'))
            data_dir.mkdir(parents=True, exist_ok=True)
        else:
            data_dir = Path(data_dir)
            assert data_dir.exists(), "{} does not exist".format(data_dir)
        data_path = data_dir / Path(system_name + "_{}/".format(int(nspins)))
        self.nspins = nspins
        self._download_dataset(data_path)
        self._hamiltonian_fn = hamiltonian_fn
        # TODO: This only works for a single order parameter. For
        #  the XY model or XYZ model, this will have to be generalized.
        self.systems = self._get_spin_systems(data_path)

    def __str__(self):
        """String representation of SpinSystemDataset object"""
        # TODO: This only works for a single order parameter. For
        #  the XY model or XYZ model, this will have to be generalized.
        return "System with {} order parameters in range [{},{}]".format(
            len(self), self.systems[0].g, self.systems[-1].g)

    def __len__(self):
        """Length of this object is determined by number of order parameters"""
        return len(self.systems)

    def _get_spin_systems(self, path):
        """Iterate over path and create `SpinSystem` for each order parameter"""
        systems = []
        for directory in [x for x in path.iterdir() if x.is_dir()]:
            g = float(str(directory.name))
            gs_energy = np.load(directory / Path('energy.npy'))[0]
            parameters = np.load(directory / Path('params.npy'))
            systems.append(
                SpinSystem(g, gs_energy, parameters, directory,
                           self._hamiltonian_fn))
        return sorted(systems)

    def _download_dataset(self, path):
        """Download the data from <location>"""
        # TODO: Implement this.


def tfi_chain(qubits, boundary_condition='closed', data_dir=None):
    """Transverse field Ising-model quantum data set.

    H = - \sum_{i} \sigma_i^z \sigma_{i+1}^z - g \sum_i \sigma_i^x

    Contains 81 circuit parameterizations corresponding to
    the ground states of the 1D TFI chain for g in [0.2,1.8].

    Example usage:

    >>> qbs = cirq.GridQubit.rect(4, 1)
    ... circuit, g, pauli_sums, systems  = tfi_chain(qbs, 'closed')
    >>> # We can print the available order parameters
    >>> # print(g)
    [0.20, 0.22, 0.24, ... ,1.76, 1.78, 1.8]
    >>> # and the circuit, corresponding to the ground state for a certain order
    >>> # parameter
    >>> print(g[10])
    0.4
    >>> print(circuit[10])
                              ┌────────────────┐
    (0, 0): ───H───ZZ──────────────────ZZ──────────Rx(0.641π)── ...
                   │                   │
    (1, 0): ───H───ZZ^0.761────ZZ──────┼───────────Rx(0.641π)── ...
                               │       │
    (2, 0): ───H───ZZ──────────ZZ^0.761┼───────────Rx(0.641π)── ...
                   │                   │
    (3, 0): ───H───ZZ^0.761────────────ZZ^0.761────Rx(0.641π)── ...
                              └────────────────┘

    Additionally, we can obtain the `cirq.PauliSum` representation of the
    Hamiltonian

    >>> print(pauli_sums[10])
    -1.000*Z((0, 0))*Z((1, 0))-1.000*Z((1, 0))*Z((2, 0))-1.000*Z((2, 0))*
    Z((3, 0))-1.000*Z((0, 0))*Z((3, 0))-0.400*X((0, 0))-0.400*X((1, 0))-
    0.400*X((2, 0))-0.400*X((3, 0))

    Finally, the fourth output, `systems`, contains additional information
    about each instance of the system (see `tfq.datasets.spin_system.SpinSystem`
    ).

    >>> # For instance, we can print the ground state obtained from
    >>> # exact diagonalization
    >>> print(systems[10].ground_state())
    [[-0.38852974+0.57092165j]
     [-0.04107317+0.06035461j]
                ...
     [-0.04107317+0.06035461j]
     [-0.38852974+0.57092165j]]
    >>> # with corresponding ground state energy
    >>> print(systems[10].gs_energy)
    -4.169142950406478
    >>> # Or we could print the parameters
    >>> print(systems[10].parameters)
    [[2.39218603 2.1284263 ]
     [2.01284773 2.30447438]]
    >>> # If we want to know the path where the data is stored we use
    >>> print(systems[10].path)
    /home/username/tfq-datasets/TFI_closed_4/0.40

    Args:
        qubits: Python `lst` of `cirq.GridQubits` of size (nspins,1).
            Supported number of spins are [4,8,12,16].
        boundary_condition: Python str` indicating the boundary condition
            of the chain. Supported boundary conditions are ['closed']
        data_dir: Location where to store the data on disk. Default is
            ~/tfq-datasets

    Returns:
        A tuple of a Python `lst` cirq.Circuit of depth N/2 with resolved
            parameters, a Python `lst` of all order parameters, a Python `lst`
            of PauliSums and a Python `lst` of SpinSystem instances
            (see Documentation).
    """

    supported_n = [4, 8, 12, 16]
    supported_bc = ['closed']

    if not all([isinstance(q, cirq.GridQubit) for q in qubits]):
        raise TypeError("qubits must be a list of cirq.GridQubit objects.")

    nspins = len(qubits)

    if not all([q.col == 0 for q in qubits]):
        raise ValueError(
            'Must be list of cirq.GridQubit objects with shape ({},0)'.format(
                nspins))

    if not len(qubits) in supported_n:
        raise ValueError('Supported number of spins are {}, received {}'.format(
            supported_n, len(qubits)))

    if boundary_condition not in supported_bc:
        raise ValueError(
            'Supported boundary conditions are {}, received {}'.format(
                supported_bc, boundary_condition))

    boundary_condition = boundary_condition
    name = 'TFI_' + boundary_condition

    def get_hamiltonian(qubits, g):
        paulisum = []
        for j in range(0, nspins - 1, 1):
            paulisum.append(-cirq.Z(qubits[j]) * cirq.Z(qubits[j + 1]))
        if boundary_condition == 'closed':
            paulisum.append(-cirq.Z(qubits[nspins - 1]) * cirq.Z(qubits[0]))
        for j in range(0, nspins):
            paulisum.append(-g * cirq.X(qubits[j]))
        hamiltonian = cirq.PauliString(paulisum[0])
        for p in paulisum[1:]:
            hamiltonian += p
        return cirq.PauliSum.from_pauli_strings(hamiltonian)

    dataset = SpinSystemDataset(nspins, name, data_dir, get_hamiltonian)

    name_generator = unique_name()
    symbols = [sympy.Symbol(next(name_generator)) for _ in range(nspins)]
    symbol_names = np.array([s.name for s in symbols]).reshape(
        (2, int(nspins / 2)))
    symbols = np.array(symbols).reshape((2, int(nspins / 2)))

    # Define the circuit
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(qubits))

    for d in range(int(nspins / 2)):
        for j in range(0, nspins - 1, 2):
            circuit.append(
                cirq.ZZ(qubits[j], qubits[j + 1])**(symbols[0, d] / np.pi))
        for j in range(1, nspins - 1, 2):
            circuit.append(
                cirq.ZZ(qubits[j], qubits[j + 1])**(symbols[0, d] / np.pi))
        if boundary_condition == 'closed':
            circuit.append(
                cirq.ZZ(qubits[nspins - 1], qubits[0])**(symbols[0, d] / np.pi))
        for j in range(0, nspins):
            circuit.append(cirq.rx(symbols[1, d])(qubits[j]))

    # Resolve the parameters
    resolved_circuits = []
    for i in range(len(dataset.systems)):
        parameter_map = dict(
            zip(symbol_names.flatten(),
                dataset.systems[i].parameters.flatten()))
        resolved_circuits.append(cirq.resolve_parameters(
            circuit, parameter_map))
    order_parameters = []
    pauli_sums = []
    for i in range(len(dataset.systems)):
        order_parameters.append(dataset.systems[i].g)
        pauli_sums.append(dataset.systems[i].hamiltonian(qubits))

    return resolved_circuits, order_parameters, pauli_sums, dataset.systems
