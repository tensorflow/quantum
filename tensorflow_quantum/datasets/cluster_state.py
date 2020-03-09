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
"""Toy dataset showing boilerplate code for a cluster state example."""
import numpy as np
import cirq


def excited_cluster_states(qubits):
    """Return a tuple of potentially excited cluster states and their labels.

    For every qubit in `qubits` this method will create a cluster state circuit
    on `qubits`, apply a `cirq.X` on that qubit along with a label of 1 and add
    it to the return dataset. Finally a cluster state circuit on `qubits` that
    doesn't contain any `cirq.X` gates with a label of -1 will be added to the
    returned dataset.

    Note: This is a toy dataset that can serve as guidance for the community
    to contribute new datasets to TensorFlow Quantum.


    >>> circuits, labels = tfq.datasets.excited_cluster_states(
    ...     cirq.GridQubit.rect(1, 3)
    ... )
    >>> print(circuits[0])
    (0, 0): ───H───@───────@───X───
                   │       │
    (0, 1): ───H───@───@───┼───────
                       │   │
    (0, 2): ───H───────@───@───────
    >>> labels[0]
    1
    >>> print(circuits[-1])
    (0, 0): ───H───@───────@───
                   │       │
    (0, 1): ───H───@───@───┼───
                       │   │
    (0, 2): ───H───────@───@───
    >>> labels[-1]
    -1


    Circuits that feature a `cirq.X` gate on one of the qubits are labeled 1,
    while the circuit that doesn't feature a `cirq.X` anywhere has the label -1.


    Args:
        qubits: Python `list` of `cirq.GridQubit`s on which the excited cluster
            state dataset will be created.

    Returns:
        A `tuple` of `cirq.Circuit`s and Python `int` labels.

    """
    if not isinstance(qubits, (tuple, list, np.ndarray)):
        raise TypeError('qubits must be a list or np.ndarray. Given: '.format(
            type(qubits)))

    for qubit in qubits:
        if not isinstance(qubit, cirq.GridQubit):
            raise ValueError('qubits must contain cirq.GridQubit only.')

    if len(qubits) <= 2:
        raise ValueError('Cluster state dataset must be defined on more than '
                         'two qubits.')

    ref_circuit = cirq.Circuit()
    ref_circuit.append(cirq.H.on_each(qubits))
    for this_bit, next_bit in zip(qubits, qubits[1:] + [qubits[0]]):
        ref_circuit.append(cirq.CZ(this_bit, next_bit))

    circuits = ()
    labels = ()

    for qubit in qubits:
        circuits += (ref_circuit + cirq.Circuit(cirq.X(qubit)),)
        labels += (1,)

    circuits += (ref_circuit,)
    labels += (-1,)

    return circuits, labels
