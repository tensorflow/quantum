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
from collections import namedtuple

import numpy as np
import sympy
import cirq


def unique_name():
    """Generator to generate an infinite number of unique names.

    Yields:
        Python `str` of the form 'theta_<integer>'

    """
    num = 0
    while True:
        yield 'theta_' + str(num)
        num += 1


def spin_system_data_set(nspins, system_name, data_dir):
    """Download data and load the data and convert to useful data structres

    Args:
        nspins: Python `int` number of spins in the system.
        system_name: Python `str` name of the system.
        data_dir: Python `str` location where to store the data on disk.
            Default is `~/tfq-datasets`

    Returns:
        A Python `lst` of Python `float` order parameters for the system
        A Python `lst` of `namedtuple` instances containing the following
            fields:
            - `g`: Numpy `float` order parameter.
            - `gs`: Numpy `complex` array ground state wave function from
                exact diagonalization.
            - `gs_energy`: Numpy `float` ground state energy from exact
                diagonalization.
            - `res_energy`: Python `float` residual between the circuit energy
                and the exact energy from exact diagonalization.
            - `fidelity`: Python `float` overlap between the circuit state
                and the exact ground state from exact diagonalization.
            - `data_path`: Python `str` location of the data set.
            - `params`: Numpy `float` array the circuit parameters with shape
                (M,circuit_depth). Here `M` is the number of parameters per
                circuit layer.

    """

    # TODO: How should downloading and storing the data be handled?
    if data_dir is None:
        data_dir = Path.expanduser(Path('~/tfq-datasets'))
        data_dir.mkdir(parents=True, exist_ok=True)
    else:
        data_dir = Path(data_dir)
        assert data_dir.exists(), "{} does not exist".format(data_dir)
    data_path = data_dir / Path(system_name + "_{}/".format(int(nspins)))
    download_dataset(system_name, data_path)
    # TODO: This only works for a single order parameter. For
    #  the XY model or XYZ model, this will have to be generalized.
    order_parameters = []
    additional_info = []
    SpinSystemInfo = namedtuple('SpinSystemInfo',
                                ['g', 'gs', 'gs_energy', 'res_energy',
                                 'fidelity', 'data_path', 'params']
                                )

    for directory in [x for x in data_path.iterdir() if x.is_dir()]:
        g = float(str(directory.name))
        with open(directory / Path('stats.txt'), 'r') as file:
            lines = file.readlines()
            res_e = float(lines[0].split('=')[1].strip('\n'))
            fidelity = float(lines[2].split('=')[1].strip('\n'))
        order_parameters.append(g)
        additional_info.append(
            SpinSystemInfo(g=g,
                           gs=np.load(directory / Path('groundstate.npy')),
                           gs_energy=np.load(directory / Path('energy.npy'))[0],
                           res_energy=res_e,
                           fidelity=fidelity,
                           data_path=str(directory),
                           params=np.load(directory / Path('params.npy')),
                           ))
    return list(zip(*sorted(zip(order_parameters, additional_info))))


def download_dataset(system_name, path):
    """
    Download the data from <location>

    Args:
        Python `str` name of the system.
        path: Python `str` location where to store the data on disk.
            Default is `~/tfq-datasets`

    """
    # TODO: Implement this.


def tfi_chain(qubits, boundary_condition='closed', data_dir=None):
    """Transverse field Ising-model quantum data set.

    $$
    H = - \sum_{i} \sigma_i^z \sigma_{i+1}^z - g \sum_i \sigma_i^x
    $$

    Contains 81 circuit parameterizations corresponding to
    the ground states of the 1D TFI chain for g in [0.2,1.8].
    This dataset contains 81 datapoints. Each datapoint is represented by a
    circuit (`cirq.Circuit`), a label (Python `float`) a hamiltonian
    (`cirq.PauliSum`) and some additional metadata. Each hamiltonian in a
    datapoint is a 1D TFI chain with boundary condition `boundary_condition` on
    `qubits` whos order parameter dictates the value of label. The circuit in a
    datapoint prepares (an approximation to) the ground state of the hamiltonian
    in the datapoint.

    Example usage:

    >>> qbs = cirq.GridQubit.rect(4, 1)
    >>> circuits, labels, pauli_sums, addinfo, variational_circuit  =
     tfq.datasets.tfi_chain(qbs, 'closed')

    We can print the available order parameters

    >>> [info.g for info in addinfo]
    [0.20, 0.22, 0.24, ... ,1.76, 1.78, 1.8]

    and the circuit corresponding to the ground state for a certain order
    parameter

    >>> labels[10]
    0
    >>> print(circuits[10])

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

    For instance, we can print the ground state obtained from
    exact diagonalization

    >>> addinfo[10].gs
    [[-0.38852974+0.57092165j]
     [-0.04107317+0.06035461j]
                ...
     [-0.04107317+0.06035461j]
     [-0.38852974+0.57092165j]]

    with corresponding ground state energy

    >>> addinfo[10].gs_energy
    -4.169142950406478

    Or we could print the parameters

    >>> addinfo[10].params
    [[2.39218603 2.1284263 ]
     [2.01284773 2.30447438]]

    If we want to know the path where the data is stored we use

    >>> addinfo[10].data_path
    /home/username/tfq-datasets/TFI_closed_4/0.40


    Args:
        qubits: Python `lst` of `cirq.GridQubit`s of size (N,1).
            Supported number of spins are [4, 8, 12, 16].
        boundary_condition: Python `str` indicating the boundary condition
            of the chain. Supported boundary conditions are ['closed']
        data_dir: Python `str` location where to store the data on disk.
            Default is `~/tfq-datasets`

    Returns:
        A Python `lst` cirq.Circuit of depth N / 2 with resolved parameters.
        A Python `lst` of labels, 0, for the ferromagnetic phase (`g<1`), 1 for
            the critical point (`g==1`) and 2 for the paramagnetic phase
            (`g>1`).
        A Python `lst` of `cirq.PauliSum`s
        A Python `lst` of `namedtuple` instances containing the following
            fields:
            - `g`: Numpy `float` order parameter.
            - `gs`: Complex numpy array ground state wave function from
                exact diagonalization.
            - `gs_energy`: Numpy `float` ground state energy from exact
                diagonalization.
            - `res_energy`: Python `float` residual between the circuit energy
                and the exact energy from exact diagonalization.
            - `fidelity`: Python `float` overlap between the circuit state
                and the exact ground state from exact diagonalization.
            - `data_path`: Python `str` location of the data set.
            - `params`: Numpy `float` array the circuit parameters with shape
                (2,circuit_depth). Here `params[0]` and `params[1]` correspond
                to the `ZZ` and `X` gate parameters respectively.
        A variational `cirq.Circuit` with unresolved parameters.
    """

    supported_n = [4, 8, 12, 16]
    supported_bc = ['closed']

    if not all(isinstance(q, cirq.GridQubit) for q in qubits):
        raise TypeError("qubits must be a list of cirq.GridQubit objects.")

    nspins = len(qubits)

    if len(qubits) not in supported_n:
        raise ValueError('Supported number of spins are {}, received {}'.format(
            supported_n, len(qubits)))

    if boundary_condition not in supported_bc:
        raise ValueError(
            'Supported boundary conditions are {}, received {}'.format(
                supported_bc, boundary_condition))

    boundary_condition = boundary_condition
    name = 'TFI_' + boundary_condition

    def get_hamiltonian(qbs, g):
        paulisum = sum(-cirq.Z(q1) * cirq.Z(q2) for q1, q2 in zip(qbs, qbs[1:]))
        if boundary_condition == 'closed':
            paulisum += -cirq.Z(qbs[0]) * cirq.Z(qbs[-1])
        paulisum += -g * sum(cirq.X(q) for q in qbs)
        return paulisum

    order_parameters, additional_info = spin_system_data_set(
        nspins, name, data_dir)

    name_generator = unique_name()
    symbols = [sympy.Symbol(next(name_generator)) for _ in range(nspins)]
    symbol_names = np.array([s.name for s in symbols]).reshape(
        (2, int(nspins / 2)))
    symbols = np.array(symbols).reshape((2, int(nspins / 2)))

    # Define the circuit
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(qubits))

    for d in range(int(nspins / 2)):
        circuit.append(
            cirq.ZZ(q1, q2) ** (symbols[0, d] / np.pi)
            for q1, q2 in zip(qubits, qubits[1:]))
        if boundary_condition == 'closed':
            circuit.append(
                cirq.ZZ(qubits[nspins - 1], qubits[0]) ** (
                        symbols[0, d] / np.pi))
        circuit.append(cirq.rx(symbols[1, d])(q1) for q1 in qubits)

    # Resolve the parameters
    resolved_circuits = []
    hamiltonians = []

    for i in range(len(order_parameters)):
        parameter_map = dict(
            zip(symbol_names.flatten(), additional_info[i].params.flatten()))
        resolved_circuits.append(cirq.resolve_parameters(
            circuit, parameter_map)
        )

        hamiltonians.append(get_hamiltonian(qubits, order_parameters[i]))

    labels = np.zeros(len(order_parameters), dtype=np.int)
    labels[np.array(order_parameters) < 1.0] = 0
    labels[np.array(order_parameters) == 1.0] = 1
    labels[np.array(order_parameters) > 1.0] = 2

    return resolved_circuits, labels, hamiltonians, additional_info, circuit
