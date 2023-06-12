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
"""Quantum datasets for quantum many-body spin systems."""

from collections import namedtuple
import os

import numpy as np
import sympy
import cirq
import tensorflow as tf

SpinSystemInfo = namedtuple(
    "SpinSystemInfo",
    [
        "g",  # Numpy `float` order parameter.
        "gs",  # Complex `np.ndarray` ground state wave function from
        # exact diagonalization.
        "gs_energy",  # Numpy `float` ground state energy from exact
        # diagonalization.
        "res_energy",  # Python `float` residual between the  circuit energy and
        # the exact energy from exact diagonalization.
        "fidelity",  # Python `float` overlap between the circuit state and the
        # exact ground state from exact diagonalization.
        "params",  # Dict with Python `str` keys and Numpy`float` values.
        # Contains $M \times P parameters. Here $M$ is the number of
        # parameters per circuit layer and $P$ the circuit depth.
        "var_circuit"  # Variational `cirq.Circuit` quantum circuit with
        # unresolved Sympy parameters.
    ])


def unique_name():
    """Generator to generate an infinite number of unique names.

    Yields:
        Python `str` of the form "theta_<integer>".

    """
    num = 0
    while True:
        yield "theta_" + str(num)
        num += 1


def _download_spin_data(system_name, boundary_condition, nspins, data_dir):
    """Download and load the data and convert to useful data structures.

    Args:
        system_name: Python `str` name of the system.
        boundary_condition: Python `str` specifying the boundary conditions of
            the system.
        nspins: Python `int` number of spins in the system.
        data_dir: Python `str` location where to store the data on disk.
            If None, the default path is `~/tfq-datasets`.If the passed
            `data_dir` does not exist, defaults to `~/tmp/.keras`.

    Returns:
        A Python `str` path where the data is stored.

    """
    # Set default storage location.
    if data_dir is None:
        data_dir = os.path.expanduser("~/tfq-datasets")

    # Use Keras file downloader.
    file_path = tf.keras.utils.get_file(
        fname=system_name + '.zip',
        cache_dir=data_dir,
        cache_subdir='spin_systems',
        origin="https://storage.googleapis.com/download"
        ".tensorflow.org/data/quantum/"
        "spin_systems/" + system_name + ".zip ",
        extract=True)

    file_path = os.path.splitext(file_path)[0]

    data_path = os.path.join(file_path, boundary_condition, str(nspins))
    return data_path


def tfi_chain(qubits, boundary_condition="closed", data_dir=None):
    """1D Transverse field Ising-model quantum data set.

    $$
    H = - \sum_{i} \sigma_i^z \sigma_{i+1}^z - g\sigma_i^x
    $$

    Contains 81 circuit parameterizations corresponding to
    the ground states of the 1D TFI chain for g in [0.2,1.8].
    This dataset contains 81 datapoints. Each datapoint is represented by a
    circuit (`cirq.Circuit`), a label (Python `float`) a Hamiltonian
    (`cirq.PauliSum`) and some additional metadata. Each Hamiltonian in a
    datapoint is a 1D TFI chain with boundary condition `boundary_condition` on
    `qubits` whos order parameter dictates the value of label. The circuit in a
    datapoint prepares (an approximation to) the ground state of the Hamiltonian
    in the datapoint.

    Example usage:

    >>> qbs = cirq.GridQubit.rect(4, 1)
    >>> circuits, labels, pauli_sums, addinfo  =
    ...     tfq.datasets.tfi_chain(qbs, "closed")

    You can print the available order parameters

    >>> [info.g for info in addinfo]
    [0.20, 0.22, 0.24, ... ,1.76, 1.78, 1.8]

    and the circuit corresponding to the ground state for a certain order
    parameter

    >>> print(circuits[10])
                                                          ┌─────── ...
    (0, 0): ───H───ZZ──────────────────────────────────ZZ───────── ...
                   │                                   │
    (1, 0): ───H───ZZ^0.761───ZZ─────────X^0.641───────┼────────── ...
                              │                        │
    (2, 0): ───H──────────────ZZ^0.761───ZZ────────────┼────────── ...
                                         │             │
    (3, 0): ───H─────────────────────────ZZ^0.761──────ZZ^0.761─── ...
                                                      └─────────── ...

    The labels indicate the phase of the system
    >>> labels[10]
    0

    Additionally, you can obtain the `cirq.PauliSum` representation of the
    Hamiltonian

    >>> print(pauli_sums[10])
    -1.000*Z((0, 0))*Z((1, 0))-1.000*Z((1, 0))*Z((2, 0))-1.000*Z((2, 0))*
    Z((3, 0))-1.000*Z((0, 0))*Z((3, 0)) ...
    -0.400*X((2, 0))-0.400*X((3, 0))

    The fourth output, `addinfo`, contains additional information
    about each instance of the system (see `tfq.datasets.spin_system.SpinSystem`
    ).

    For instance, you can print the ground state obtained from
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

    You can also inspect the parameters

    >>> addinfo[10].params
    {"theta_0": 0.7614564630036476, "theta_1": 0.6774991338794768,
    "theta_2": 0.6407093304791429, "theta_3": 0.7335369771742435}

    and change them to experiment with different parameter values by using
    the unresolved variational circuit returned by tfichain
    >>> new_params = {}
    ... for symbol_name, value in addinfo[10].params.items():
    ...    new_params[symbol_name] = 0.5 * value
    >>> new_params
    {"theta_0": 0.3807282315018238, "theta_1": 0.3387495669397384,
    "theta_2": 0.32035466523957146, "theta_3": 0.36676848858712174}
    >>> new_circuit = cirq.resolve_parameters(addinfo[10].var_circuit,
    ... new_params)
    >>> print(new_circuit)
                                                           ┌─────── ...
    (0, 0): ───H───ZZ──────────────────────────────────ZZ───────── ...
                   │                                   │
    (1, 0): ───H───ZZ^0.761───ZZ─────────X^0.32────────┼────────── ...
                              │                        │
    (2, 0): ───H──────────────ZZ^0.761───ZZ────────────┼────────── ...
                                         │             │
    (3, 0): ───H─────────────────────────ZZ^0.761──────ZZ^0.761─── ...
                                                      └─────────── ...

    Args:
        qubits: Python `lst` of `cirq.GridQubit`s. Supported number of spins
            are [4, 8, 12, 16].
        boundary_condition: Python `str` indicating the boundary condition
            of the chain. Supported boundary conditions are ["closed"].
        data_dir: Optional Python `str` location where to store the data on
            disk. Defaults to `/tmp/.keras`.
    Returns:
        A Python `lst` cirq.Circuit of depth len(qubits) / 2 with resolved
            parameters.
        A Python `lst` of labels, 0, for the ferromagnetic phase (`g<1`), 1 for
            the critical point (`g==1`) and 2 for the paramagnetic phase
            (`g>1`).
        A Python `lst` of `cirq.PauliSum`s.
        A Python `lst` of `namedtuple` instances containing the following
            fields:
        - `g`: Numpy `float` order parameter.
        - `gs`: Complex `np.ndarray` ground state wave function from
            exact diagonalization.
        - `gs_energy`: Numpy `float` ground state energy from exact
            diagonalization.
        - `res_energy`: Python `float` residual between the circuit energy
            and the exact energy from exact diagonalization.
        - `fidelity`: Python `float` overlap between the circuit state
            and the exact ground state from exact diagonalization.
        - `params`: Dict with Python `str` keys and Numpy`float` values.
            Contains $M \times P $ parameters. Here $M$ is the number of
            parameters per circuit layer and $P$ the circuit depth.
        - `var_circuit`: Variational `cirq.Circuit` quantum circuit with
            unresolved Sympy parameters.
    """

    supported_n = [4, 8, 12, 16]
    supported_bc = ["closed"]
    if any(isinstance(q, list) for q in qubits):
        raise TypeError("qubits must be a one-dimensional list")

    if not all(isinstance(q, cirq.GridQubit) for q in qubits):
        raise TypeError("qubits must be a list of cirq.GridQubit objects.")

    nspins = len(qubits)
    depth = nspins // 2
    if nspins not in supported_n:
        raise ValueError("Supported number of spins are {}, received {}".format(
            supported_n, nspins))

    if boundary_condition not in supported_bc:
        raise ValueError(
            "Supported boundary conditions are {}, received {}".format(
                supported_bc, boundary_condition))

    data_path = _download_spin_data('TFI_chain', boundary_condition, nspins,
                                    data_dir)

    name_generator = unique_name()

    # 2 * N/2 parameters.
    symbol_names = [next(name_generator) for _ in range(nspins)]
    symbols = [sympy.Symbol(name) for name in symbol_names]

    # Define the circuit.
    circuit = cirq.Circuit(cirq.H.on_each(qubits))

    for d in range(depth):
        circuit.append(
            cirq.ZZ(q1, q2)**(symbols[d]) for q1, q2 in zip(qubits, qubits[1:]))
        if boundary_condition == "closed":
            circuit.append(cirq.ZZ(qubits[nspins - 1], qubits[0])**(symbols[d]))
        circuit.append(cirq.X(q1)**(symbols[d + depth]) for q1 in qubits)

    # Initiate lists.
    resolved_circuits = []
    hamiltonians = []
    order_parameters = []
    additional_info = []
    labels = []
    # Load the data and append to the lists.
    for i, directory in enumerate(x for x in os.listdir(data_path)):
        # The folders are named according to the order value data they contain.
        g = float(directory)
        with open(os.path.join(data_path, directory, "stats.txt"), "r") as file:
            lines = file.readlines()
            res_e = float(lines[0].split("=")[1].strip("\n"))
            fidelity = float(lines[2].split("=")[1].strip("\n"))
        order_parameters.append(g)
        params = np.load(os.path.join(data_path, directory, "params.npy")) \
                 / np.pi
        # Parameters are stored as np.float32, but cirq expects np.float64
        # See https://github.com/quantumlib/Cirq/issues/3359
        params = params.astype(np.float)
        additional_info.append(
            SpinSystemInfo(g=g,
                           gs=np.load(
                               os.path.join(data_path, directory,
                                            "groundstate.npy"))[:, 0],
                           gs_energy=np.load(
                               os.path.join(data_path, directory,
                                            "energy.npy"))[0],
                           res_energy=res_e,
                           fidelity=fidelity,
                           params=dict(zip(symbol_names, params.flatten())),
                           var_circuit=circuit))

        # Resolve the circuit parameters.
        resolved_circuit = cirq.resolve_parameters(circuit,
                                                   additional_info[i].params)
        resolved_circuits.append(resolved_circuit)

        # Make the PauliSum.
        paulisum = sum(
            -cirq.Z(q1) * cirq.Z(q2) for q1, q2 in zip(qubits, qubits[1:]))
        if boundary_condition == "closed":
            paulisum += -cirq.Z(qubits[0]) * cirq.Z(qubits[-1])
        paulisum += -order_parameters[i] * sum(cirq.X(q) for q in qubits)
        hamiltonians.append(paulisum)

        # Set labels for the different phases.
        if order_parameters[i] < 1.0:
            labels.append(0)
        elif order_parameters[i] == 1.0:
            labels.append(1)
        else:
            labels.append(2)
    # Make sure that the data is ordered from g=0.2 to g=1.8.
    _, resolved_circuits, labels, hamiltonians, additional_info = zip(*sorted(
        zip(order_parameters, resolved_circuits, labels, hamiltonians,
            additional_info)))

    return resolved_circuits, labels, hamiltonians, additional_info


def xxz_chain(qubits, boundary_condition="closed", data_dir=None):
    """1D XXZ model quantum data set.

    $$
    H = \sum_{i} \sigma_i^x \sigma_{i+1}^x + \sigma_i^y \sigma_{i+1}^y +
    \Delta\sigma_i^z \sigma_{i+1}^z
    $$

    Contains 76 circuit parameterizations corresponding to
    the ground states of the 1D XXZ chain for g in [0.3,1.8].
    This dataset contains 76 datapoints. Each datapoint is represented by a
    circuit (`cirq.Circuit`), a label (Python `float`) a Hamiltonian
    (`cirq.PauliSum`) and some additional metadata. Each Hamiltonian in a
    datapoint is a 1D XXZ chain with boundary condition `boundary_condition` on
    `qubits` whos order parameter dictates the value of label. The circuit in a
    datapoint prepares (an approximation to) the ground state of the Hamiltonian
    in the datapoint.

    Example usage:

    >>> qbs = cirq.GridQubit.rect(4, 1)
    >>> circuits, labels, pauli_sums, addinfo  =
    ...     tfq.datasets.xxz_chain(qbs, "closed")

    You can print the available order parameters

    >>> [info.g for info in addinfo]
    [0.30, 0.32, 0.34, ... ,1.76, 1.78, 1.8]

    and the circuit corresponding to the ground state for a certain order
    parameter

    >>> print(circuits[10])
                           ┌──────────────────┐   ┌──────────────────┐
    (0, 0): ───X───H───@─────────────ZZ─────────────────────YY────────── ...
                       │             │                      │
    (1, 0): ───X───────X────ZZ───────┼─────────────YY───────┼─────────── ...
                            │        │             │        │
    (2, 0): ───X───H───@────ZZ^-0.922┼─────────────YY^-0.915┼─────────── ...
                       │             │                      │
    (3, 0): ───X───────X─────────────ZZ^-0.922──────────────YY^-0.915─── ...
                           └──────────────────┘   └──────────────────┘

    The labels indicate the phase of the system
    >>> labels[10]
    0

    Additionally, you can obtain the `cirq.PauliSum` representation of the
    Hamiltonian

    >>> print(pauli_sums[10])
    0.400*Z((0, 0))*Z((1, 0))+0.400*Z((1, 0))*Z((2, 0))+ ...
    +1.000*Y((0, 0))*Y((3, 0))+1.000*X((0, 0))*X((3, 0))

    The fourth output, `addinfo`, contains additional information
    about each instance of the system (see `tfq.datasets.spin_system.SpinSystem`
    ).

    For instance, you can print the ground state obtained from
    exact diagonalization

    >>> addinfo[10].gs
    [-8.69032854e-18-6.58023246e-20j  4.54546402e-17+3.08736567e-17j
     -9.51026525e-18+2.42638062e-17j  4.52284042e-02+3.18111120e-01j
                                    ...
      4.52284042e-02+3.18111120e-01j -6.57974275e-18-3.84526414e-17j
     -1.60673943e-17+5.79767820e-17j  2.86193021e-17-5.06694574e-17j]

    with corresponding ground state energy

    >>> addinfo[10].gs_energy
    -6.744562646538039

    You can also inspect the parameters

    >>> addinfo[10].params
    {'theta_0': 1.0780547, 'theta_1': 0.99271035, 'theta_2': 1.0854135, ...

    and change them to experiment with different parameter values by using
    the unresolved variational circuit returned by xxzchain
    >>> new_params = {}
    ... for symbol_name, value in addinfo[10].params.items():
    ...    new_params[symbol_name] = 0.5 * value
    >>> new_params
    {'theta_0': 0.5390273332595825, 'theta_1': 0.49635517597198486, ...
    >>> new_circuit = cirq.resolve_parameters(addinfo[10].var_circuit,
    ... new_params)
    >>> print(new_circuit)
                           ┌──────────────────┐   ┌──────────────────┐
    (0, 0): ───X───H───@─────────────ZZ─────────────────────YY────────── ...
                       │             │                      │
    (1, 0): ───X───────X────ZZ───────┼─────────────YY───────┼─────────── ...
                            │        │             │        │
    (2, 0): ───X───H───@────ZZ^(7/13)┼─────────────YY^0.543 ┼─────────── ...
                       │             │                      │
    (3, 0): ───X───────X─────────────ZZ^(7/13)──────────────YY^0.543 ─── ...
                           └──────────────────┘   └──────────────────┘
    Args:
        qubits: Python `lst` of `cirq.GridQubit`s. Supported number of spins
            are [4, 8, 12, 16].
        boundary_condition: Python `str` indicating the boundary condition
            of the chain. Supported boundary conditions are ["closed"].
        data_dir: Optional Python `str` location where to store the data on
            disk. Defaults to `/tmp/.keras`.
    Returns:
        A Python `lst` cirq.Circuit of depth len(qubits) / 2 with resolved
            parameters.
        A Python `lst` of labels, 0, for the critical metallic phase
            (`Delta<=1`) and 1 for the insulating phase (`Delta>1`).
        A Python `lst` of `cirq.PauliSum`s.
        A Python `lst` of `namedtuple` instances containing the following
            fields:
        - `g`: Numpy `float` order parameter.
        - `gs`: Complex `np.ndarray` ground state wave function from
            exact diagonalization.
        - `gs_energy`: Numpy `float` ground state energy from exact
            diagonalization.
        - `res_energy`: Python `float` residual between the circuit energy
            and the exact energy from exact diagonalization.
        - `fidelity`: Python `float` overlap between the circuit state
            and the exact ground state from exact diagonalization.
        - `params`: Dict with Python `str` keys and Numpy`float` values.
            Contains $M \times P $ parameters. Here $M$ is the number of
            parameters per circuit layer and $P$ the circuit depth.
        - `var_circuit`: Variational `cirq.Circuit` quantum circuit with
            unresolved Sympy parameters.
    """

    supported_n = [4, 8, 12, 16]
    supported_bc = ["closed"]
    if any(isinstance(q, list) for q in qubits):
        raise TypeError("qubits must be a one-dimensional list")

    if not all(isinstance(q, cirq.GridQubit) for q in qubits):
        raise TypeError("qubits must be a list of cirq.GridQubit objects.")

    nspins = len(qubits)
    depth = nspins // 2
    if nspins not in supported_n:
        raise ValueError("Supported number of spins are {}, received {}".format(
            supported_n, nspins))

    if boundary_condition not in supported_bc:
        raise ValueError(
            "Supported boundary conditions are {}, received {}".format(
                supported_bc, boundary_condition))

    data_path = _download_spin_data('XXZ_chain', boundary_condition, nspins,
                                    data_dir)

    name_generator = unique_name()

    # 4 * N/2 parameters.
    symbol_names = [next(name_generator) for _ in range(2 * nspins)]
    symbols = [sympy.Symbol(name) for name in symbol_names]

    # Define the circuit.
    circuit = cirq.Circuit(cirq.X.on_each(qubits))
    even_qubits = qubits[::2]
    odd_qubits = qubits[1::2]
    circuit.append(cirq.H(qubits[i]) for i in range(0, nspins, 2))
    circuit.append(cirq.CNOT(q1, q2) for q1, q2 in zip(even_qubits, odd_qubits))

    for d in range(depth):
        for q1, q2 in zip(odd_qubits, even_qubits[1:]):
            circuit.append(cirq.ZZ(q1, q2)**(symbols[d]))
            circuit.append(cirq.YY(q1, q2)**(symbols[d + depth]))
            circuit.append(cirq.XX(q1, q2)**(symbols[d + depth]))
        if boundary_condition == "closed":
            circuit.append(cirq.ZZ(qubits[-1], qubits[0])**(symbols[d]))
            circuit.append(cirq.YY(qubits[-1], qubits[0])**(symbols[d + depth]))
            circuit.append(cirq.XX(qubits[-1], qubits[0])**(symbols[d + depth]))
        for q1, q2 in zip(even_qubits, odd_qubits):
            circuit.append(cirq.ZZ(q1, q2)**(symbols[d + 2 * depth]))
            circuit.append(cirq.YY(q1, q2)**(symbols[d + 3 * depth]))
            circuit.append(cirq.XX(q1, q2)**(symbols[d + 3 * depth]))
    # Initiate lists.
    resolved_circuits = []
    hamiltonians = []
    order_parameters = []
    additional_info = []
    labels = []
    # Load the data and append to the lists.
    for i, directory in enumerate(x for x in os.listdir(data_path)):
        # The folders are named according to the order value data they contain.
        g = float(directory)
        with open(os.path.join(data_path, directory, "stats.txt"), "r") as file:
            lines = file.readlines()
            res_e = float(lines[0].split("=")[1].strip("\n"))
            fidelity = float(lines[2].split("=")[1].strip("\n"))
        order_parameters.append(g)
        params = np.load(os.path.join(data_path, directory, "params.npy")) \
                 / np.pi
        # Parameters are stored as np.float32, but cirq expects np.float64
        # See https://github.com/quantumlib/Cirq/issues/3359
        params = params.astype(np.float)
        additional_info.append(
            SpinSystemInfo(g=g,
                           gs=np.load(
                               os.path.join(data_path, directory,
                                            "groundstate.npy"))[:, 0],
                           gs_energy=np.load(
                               os.path.join(data_path, directory,
                                            "energy.npy"))[0],
                           res_energy=res_e,
                           fidelity=fidelity,
                           params=dict(zip(symbol_names, params.flatten())),
                           var_circuit=circuit))
        # Resolve the circuit parameters.
        resolved_circuit = cirq.resolve_parameters(circuit,
                                                   additional_info[i].params)
        resolved_circuits.append(resolved_circuit)
        # Make the PauliSum.
        paulisum = sum(order_parameters[i] * cirq.Z(q1) * cirq.Z(q2) +
                       cirq.Y(q1) * cirq.Y(q2) + cirq.X(q1) * cirq.X(q2)
                       for q1, q2 in zip(qubits, qubits[1:]))

        if boundary_condition == "closed":
            paulisum += order_parameters[i] * cirq.Z(qubits[0]) * cirq.Z(
                qubits[-1]) + cirq.Y(qubits[0]) * cirq.Y(qubits[-1]) + cirq.X(
                    qubits[0]) * cirq.X(qubits[-1])
        hamiltonians.append(paulisum)

        # Set labels for the different phases.
        if order_parameters[i] <= 1.0:
            labels.append(0)
        else:
            labels.append(1)

    # Make sure that the data is ordered from g=0.2 to g=1.8.
    _, resolved_circuits, labels, hamiltonians, additional_info = zip(*sorted(
        zip(order_parameters, resolved_circuits, labels, hamiltonians,
            additional_info)))

    return resolved_circuits, labels, hamiltonians, additional_info


def tfi_rectangular(qubits, boundary_condition="torus", data_dir=None):
    """2D transverse field Ising-model quantum data set.

    $$
    H = - \sum_{\langle i,j \rangle} \sigma_i^z \sigma_{j}^z - g\sigma_i^x
    $$

    Contains 51 circuit parameterizations corresponding to
    the ground states of the 2D TFI chain for g in [2.5,3.5].
    This dataset contains 51 datapoints. Each datapoint is represented by a
    circuit (`cirq.Circuit`), a label (Python `float`) a Hamiltonian
    (`cirq.PauliSum`) and some additional metadata. Each Hamiltonian in a
    datapoint is a 2D TFI rectangular lattice with boundary condition
    `boundary_condition` on `qubits` whos order parameter dictates the value of
    label. The circuit in a datapoint prepares (an approximation to) the ground
    state of the Hamiltonian in the datapoint.

    Example usage:

    >>> qbs = cirq.GridQubit.rect(9, 1)
    >>> circuits, labels, pauli_sums, addinfo  =
    ...     tfq.datasets.tfi_rectangular(qbs, "torus")

    You can print the available order parameters

    >>> [info.g for info in addinfo]
    [2.5, 2.52, 2.54, ... ,3.46 , 3.48, 3.5]

    and the circuit corresponding to the ground state for a certain order
    parameter

    >>> print(circuits[10])
                       ┌──────────────────────┐   ┌───────────────────── ...
    (0, 0): ───H────ZZ─────────────────────────ZZ─────────────────────── ...
                    │                          │
    (1, 0): ───H────ZZ^0.948896────────────────┼──────────ZZ──────────── ...
                                               │          │
    (2, 0): ───H────ZZ─────────────────────────┼──────────┼───────────── ...
                    │                          │          │
    (3, 0): ───H────┼──────────ZZ──────────────┼──────────┼───────────── ...
       .            .                          .          .
       .            .                          .          .

    The labels indicate the phase of the system
    >>> labels[10]
    0

    Additionally, you can obtain the `cirq.PauliSum` representation of the
    Hamiltonian

    >>> print(pauli_sums[10])
    -2.700*X((0, 0))-2.700*X((1, 0))-2.700*X((2, 0))-2.700*X((3, 0))-
    2.700*X((4, 0))-2.700*X((5, 0))-2.700*X((6, 0))-2.700*X((7, 0))- ...
    -1.000*Z((3, 0))*Z((6, 0))-1.000*Z((4, 0))*Z((5, 0))

    The fourth output, `addinfo`, contains additional information
    about each instance of the system (see `tfq.datasets.spin_system.SpinSystem`
    ).

    For instance, you can print the ground state obtained from
    exact diagonalization

    >>> addinfo[10].gs
    [-0.11843355-0.30690906j -0.04374221-0.11335368j -0.04374221-0.11335368j
     -0.02221491-0.0575678j  -0.04374221-0.11335368j -0.02221491-0.0575678j
                ...
     -0.04374221-0.11335368j -0.02221491-0.0575678j  -0.04374221-0.11335368j
     -0.04374221-0.11335368j -0.11843355-0.30690906j]

    with corresponding ground state energy

    >>> addinfo[10].gs_energy
    -26.974953331962762

    You can also inspect the parameters

    >>> addinfo[10].params
    {'theta_0': 0.948896, 'theta_1': 0.90053445, ...
    'theta_8': 0.76966083, 'theta_9': 0.87608284}

    and change them to experiment with different parameter values by using
    the unresolved variational circuit returned by tfichain
    >>> new_params = {}
    ... for symbol_name, value in addinfo[10].params.items():
    ...    new_params[symbol_name] = 0.5 * value
    >>> new_params
    {'theta_0': 0.47444799542427063, 'theta_1': 0.4502672255039215, ...
    'theta_8': 0.38483041524887085, 'theta_9': 0.43804141879081726}
    >>> new_circuit = cirq.resolve_parameters(addinfo[10].var_circuit,
    ... new_params)
    >>> print(new_circuit)
                       ┌──────────────────────┐   ┌───────────────────── ...
    (0, 0): ───H────ZZ─────────────────────────ZZ─────────────────────── ...
                    │                          │
    (1, 0): ───H────ZZ^0.474───────────────────┼──────────ZZ──────────── ...
                                               │          │
    (2, 0): ───H────ZZ─────────────────────────┼──────────┼───────────── ...
                    │                          │          │
    (3, 0): ───H────┼──────────ZZ──────────────┼──────────┼───────────── ...
       .            .                          .          .
       .            .                          .          .

    Args:
        qubits: Python `lst` of `cirq.GridQubit`s. Supported number of spins
            are [9, 12, 16].
        boundary_condition: Python `str` indicating the boundary condition
            of the chain. Supported boundary conditions are ["torus"].
        data_dir: Optional Python `str` location where to store the data on
            disk. Defaults to `/tmp/.keras`.
    Returns:
        A Python `lst` cirq.Circuit of depth ceil(len(qubits) / 2) with resolved
            parameters.
        A Python `lst` of labels, 0, for the phase (`g<3.04`),
            1 for the critical point (`g==3.04`) and 2 for the phase (`g>3.04`).
        A Python `lst` of `cirq.PauliSum`s.
        A Python `lst` of `namedtuple` instances containing the following
            fields:
        - `g`: Numpy `float` order parameter.
        - `gs`: Complex `np.ndarray` ground state wave function from
            exact diagonalization.
        - `gs_energy`: Numpy `float` ground state energy from exact
            diagonalization.
        - `res_energy`: Python `float` residual between the circuit energy
            and the exact energy from exact diagonalization.
        - `fidelity`: Python `float` overlap between the circuit state
            and the exact ground state from exact diagonalization.
        - `params`: Dict with Python `str` keys and Numpy`float` values.
            Contains $M \times P $ parameters. Here $M$ is the number of
            parameters per circuit layer and $P$ the circuit depth.
        - `var_circuit`: Variational `cirq.Circuit` quantum circuit with
            unresolved Sympy parameters.
    """

    supported_n = [9, 12, 16]
    supported_bc = ["torus"]
    if any(isinstance(q, list) for q in qubits):
        raise TypeError("qubits must be a one-dimensional list")

    if not all(isinstance(q, cirq.GridQubit) for q in qubits):
        raise TypeError("qubits must be a list of cirq.GridQubit objects.")

    nspins = len(qubits)
    depth = int(np.ceil(nspins / 2))

    if nspins not in supported_n:
        raise ValueError("Supported number of spins are {}, received {}".format(
            supported_n, nspins))

    if boundary_condition not in supported_bc:
        raise ValueError(
            "Supported boundary conditions are {}, received {}".format(
                supported_bc, boundary_condition))

    data_path = _download_spin_data('TFI_rect', boundary_condition, nspins,
                                    data_dir)

    name_generator = unique_name()

    # 2 * depth parameters.
    symbol_names = [next(name_generator) for _ in range(2 * depth)]

    symbols = [sympy.Symbol(name) for name in symbol_names]

    # Define the circuit.
    circuit = cirq.Circuit(cirq.H.on_each(qubits))
    if boundary_condition == 'torus':
        if nspins == 9:
            #3x3
            edges = {
                'g1': [[0, 1], [2, 5], [3, 4], [6, 7]],
                'g2': [[0, 6], [1, 4], [7, 8]],
                'g3': [[0, 3], [1, 2], [4, 7], [5, 8]],
                'g4': [[0, 2], [1, 7], [3, 5], [6, 8]],
                'g5': [[2, 8], [3, 6], [4, 5]]
            }
        if nspins == 12:
            #4x3
            edges = {
                'g1': [[0, 3], [1, 2], [4, 7], [5, 8], [6, 9], [10, 11]],
                'g2': [[0, 1], [2, 5], [3, 4], [6, 7], [8, 11], [9, 10]],
                'g3': [[0, 9], [1, 10], [2, 11], [3, 6], [4, 5], [7, 8]],
                'g4': [[0, 2], [1, 4], [3, 5], [6, 8], [7, 10], [9, 11]]
            }
        if nspins == 16:
            #4x4
            edges = {
                'g1': [[0, 3], [1, 2], [4, 7], [5, 9], [6, 10], [8, 12],
                       [11, 15], [13, 14]],
                'g2': [[0, 4], [1, 5], [2, 6], [3, 15], [7, 11], [8, 9],
                       [10, 14], [12, 13]],
                'g3': [[0, 12], [1, 13], [2, 3], [4, 5], [6, 7], [8, 11],
                       [9, 10], [14, 15]],
                'g4': [[0, 1], [2, 14], [3, 7], [4, 8], [5, 6], [9, 13],
                       [10, 11], [12, 15]]
            }
    for d in range(depth):
        for graph in edges.values():
            circuit.append(
                cirq.ZZ(qubits[edge[0]], qubits[edge[1]])**(symbols[d])
                for edge in graph)
        circuit.append(cirq.X(q1)**(symbols[d + depth]) for q1 in qubits)

    # Initiate lists.
    resolved_circuits = []
    hamiltonians = []
    order_parameters = []
    additional_info = []
    labels = []

    # Load the data and append to the lists.
    for i, directory in enumerate(x for x in os.listdir(data_path)):
        # The folders are named according to the order value data they contain.
        g = float(directory)
        with open(os.path.join(data_path, directory, "stats.txt"), "r") as file:
            lines = file.readlines()
            res_e = float(lines[0].split("=")[1].strip("\n"))
            fidelity = float(lines[2].split("=")[1].strip("\n"))
        order_parameters.append(g)
        params = np.load(os.path.join(data_path, directory, "params.npy")) \
                 / np.pi
        additional_info.append(
            SpinSystemInfo(g=g,
                           gs=np.load(
                               os.path.join(data_path, directory,
                                            "groundstate.npy"))[:, 0],
                           gs_energy=np.load(
                               os.path.join(data_path, directory,
                                            "energy.npy"))[0],
                           res_energy=res_e,
                           fidelity=fidelity,
                           params=dict(zip(symbol_names, params.flatten())),
                           var_circuit=circuit))
        # Resolve the circuit parameters.
        param_resolver = cirq.resolve_parameters(circuit,
                                                 additional_info[i].params)
        resolved_circuits.append(param_resolver)
        paulisum = -order_parameters[i] * sum(cirq.X(q) for q in qubits)
        # Make the PauliSum.
        for graph in edges.values():
            paulisum -= sum(
                cirq.Z(qubits[edge[0]]) * cirq.Z(qubits[edge[1]])
                for edge in graph)

        hamiltonians.append(paulisum)

        # Set labels for the different phases.
        if order_parameters[i] < 3.04:
            labels.append(0)
        elif order_parameters[i] == 3.04:
            labels.append(1)
        else:
            labels.append(2)

    # Make sure that the data is ordered from g=2.5 to g=3.5.
    _, resolved_circuits, labels, hamiltonians, additional_info = zip(*sorted(
        zip(order_parameters, resolved_circuits, labels, hamiltonians,
            additional_info)))

    return resolved_circuits, labels, hamiltonians, additional_info
