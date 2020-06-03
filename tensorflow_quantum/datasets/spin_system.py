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

from collections import namedtuple
import os

import numpy as np
import sympy
import cirq
from tensorflow.keras.utils import get_file

SpinSystemInfo = namedtuple(
    "SpinSystemInfo",
    [
        "g",  # Numpy `float` order parameter.
        "gs",  # Numpy `complex` array ground state wave function from exact
        # diagonalization.
        "gs_energy",  # Numpy `float` ground state energy from exact
        # diagonalization.
        "res_energy",  # Python `float` residual between the  circuit energy and
        # the exact energy from exact diagonalization.
        "fidelity",  # Python `float` overlap between the circuit state and the
        # exact ground state from exact diagonalization.
        "params"  # Dict with Python `str` keys and Numpy`float` values.
        # Contains $M \times P parameters. Here $M$ is the number of
        # parameters per circuit layer and $P$ the circuit depth.
    ])


def unique_name():
    """Generator to generate an infinite number of unique names.

    Yields:
        Python `str` of the form "theta_<integer>"

    """
    num = 0
    while True:
        yield "theta_" + str(num)
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
        - `params`: Dict with Python `str` keys and Numpy`float` values.
            Contains $M \times P $ parameters. Here $M$ is the number of
            parameters per circuit layer and $P$ the circuit depth.

    """
    # Set default storage location
    if data_dir is None:
        data_dir = os.path.expanduser("~/tfq-datasets")
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

    # Use Keras file downloader.
    get_file(fname='spin_systems',
             cache_dir=data_dir,
             cache_subdir='',
             origin="https://storage.googleapis.com/"
             "download.tensorflow.org/data/"
             "quantum/spin_systems_data.zip",
             extract=True)
    data_path = data_dir + "/tfq_data-master/spin_systems/" + system_name \
                + "_{}".format(nspins)
    # TODO: This only works for a single order parameter. For
    #  the XY model or XYZ model, this will have to be generalized.
    order_parameters = []
    additional_info = []
    parameters = []
    for directory in [x for x in os.listdir(data_path)]:
        g = float(directory)
        with open(data_path + "/" + directory + "/stats.txt", "r") as file:
            lines = file.readlines()
            res_e = float(lines[0].split("=")[1].strip("\n"))
            fidelity = float(lines[2].split("=")[1].strip("\n"))
        order_parameters.append(g)
        additional_info.append(
            SpinSystemInfo(
                g=g,
                gs=np.load(data_path + "/" + directory +
                           "/groundstate.npy")[:, 0],
                gs_energy=np.load(data_path + "/" + directory +
                                  "/energy.npy")[0],
                res_energy=res_e,
                fidelity=fidelity,
                params={},
            ))
        parameters.append(
            np.load(data_path + "/" + directory + "/params.npy") / np.pi)
    return list(
        zip(*sorted(zip(order_parameters, additional_info, parameters))))


def tfi_chain(qubits, boundary_condition="closed", data_dir=None):
    """Transverse field Ising-model quantum data set.

    $$
    H = - \sum_{i} \sigma_i^z \sigma_{i+1}^z - g \sum_i \sigma_i^x
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
    >>> circuits, labels, pauli_sums, addinfo, variational_circuit  =
    ... tfq.datasets.tfi_chain(qbs, "closed")

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
    Z((3, 0))-1.000*Z((0, 0))*Z((3, 0))-0.400*X((0, 0))-0.400*X((1, 0))-
    0.400*X((2, 0))-0.400*X((3, 0))

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

    # and change them to experiment with different parameter values by using
    # the unresolved variational circuit returned by tfichain
    >>> new_params = {}
    ... for symbol_name, value in addinfo[10].params.items():
    ...    new_params[symbol_name] = 0.5 * value
    >>> new_params
    {"theta_0": 0.3807282315018238, "theta_1": 0.3387495669397384,
    "theta_2": 0.32035466523957146, "theta_3": 0.36676848858712174}
    >>> new_circuit = cirq.resolve_parameters(variational_circuit, new_params)
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
            of the chain. Supported boundary conditions are ["closed"]
        data_dir: Optional Python `str` location where to store the data on
            disk. Defaults to `~/tfq-datasets`
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
        - `params`: Python `dict` with a field `raw` containing a Numpy
            `float` array with circuit parameters with shape
            (2, circuit_depth). Here `params[0]` and `params[1]` correspond
            to the `ZZ` and `X` gate parameters respectively. The field
            `symbol_names` contains a Numpy `str` array with a unique symbol
            name for each parameter. These names correspond to the `Sympy`
            variables in the returned variational circuit. The field
            `param_resolver`
        A variational `cirq.Circuit` with unresolved parameters.
    """

    supported_n = [4, 8, 12, 16]
    supported_bc = ["closed"]
    if not all(not isinstance(q, list) for q in qubits):
        raise TypeError("qubits must be a one-dimensional list")

    if not all(isinstance(q, cirq.GridQubit) for q in qubits):
        raise TypeError("qubits must be a list of cirq.GridQubit objects.")

    nspins = len(qubits)

    if nspins not in supported_n:
        raise ValueError("Supported number of spins are {}, received {}".format(
            supported_n, len(qubits)))

    if boundary_condition not in supported_bc:
        raise ValueError(
            "Supported boundary conditions are {}, received {}".format(
                supported_bc, boundary_condition))

    name = "TFI_" + boundary_condition

    order_parameters, additional_info, params = spin_system_data_set(
        nspins, name, data_dir)

    name_generator = unique_name()

    # 2 * N/2 parameters
    symbols = [sympy.Symbol(next(name_generator)) for _ in range(nspins)]
    symbol_names = np.array([s.name for s in symbols]).reshape(
        (2, int(nspins / 2)))
    symbols = np.array(symbols).reshape((2, nspins // 2))

    # Define the circuit
    circuit = cirq.Circuit(cirq.H.on_each(qubits))

    for d in range(nspins // 2):
        circuit.append(
            cirq.ZZ(q1, q2)**(symbols[0, d])
            for q1, q2 in zip(qubits, qubits[1:]))
        if boundary_condition == "closed":
            circuit.append(
                cirq.ZZ(qubits[nspins - 1], qubits[0])**(symbols[0, d]))
        circuit.append(cirq.X(q1)**(symbols[1, d]) for q1 in qubits)

    # Resolve the parameters
    resolved_circuits = []
    hamiltonians = []

    for i in range(len(order_parameters)):
        for symbol_name, value in zip(symbol_names.flatten(),
                                      params[i].flatten()):
            additional_info[i].params[symbol_name] = value
        param_resolver = cirq.resolve_parameters(circuit,
                                                 additional_info[i].params)
        resolved_circuits.append(param_resolver)

        # Make the PauliSum
        paulisum = sum(
            -cirq.Z(q1) * cirq.Z(q2) for q1, q2 in zip(qubits, qubits[1:]))
        if boundary_condition == "closed":
            paulisum += -cirq.Z(qubits[0]) * cirq.Z(qubits[-1])
        paulisum += -order_parameters[i] * sum(cirq.X(q) for q in qubits)
        hamiltonians.append(paulisum)

    labels = np.zeros(len(order_parameters), dtype=np.int)
    labels[np.array(order_parameters) < 1.0] = 0
    labels[np.array(order_parameters) == 1.0] = 1
    labels[np.array(order_parameters) > 1.0] = 2

    return resolved_circuits, labels, hamiltonians, additional_info, circuit
