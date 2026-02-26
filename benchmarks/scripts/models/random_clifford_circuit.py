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
"""Module for generating random Clifford circuits."""

from typing import Iterable

import numpy as np
import cirq


def random_clifford_circuit(qubits, n_moments, op_density, random_state=None):
    """Generate a dense circuit using elements of C2.

    Each layer will consist of a random number of one- or two-qubit Clifford
    gates acting on a random subset of qubits.
    Args:
        qubits: The sequence of GridQubits that the circuit should act on.
            Because the qubits on which an operation acts are chosen randomly,
            not all given qubits may be acted upon.
        n_moments: The number of moments in the generated circuit.
        op_density: the expected fraction of qubits acted on in each
            moment in half-open interval [0, 1].
        random_state: Optional random state or random state seed.

    Returns:
        Clifford circuit with randomly chosen and assigned gates.
    """
    if random_state and not isinstance(random_state,
                                       (np.random.RandomState, int)):
        raise TypeError("Random state input must be a numpy RandomState or an "
                        "integer seed to a random state.")

    if not isinstance(qubits, Iterable) or not all(
            isinstance(q, cirq.GridQubit) for q in qubits):
        raise TypeError("Must provide an iterable of GridQubits.")

    n_qubits = len(qubits)
    if n_qubits < 2:
        raise ValueError("Must provide at least 2 qubits to circuit generator.")

    rng = np.random
    if isinstance(random_state, np.random.RandomState):
        rng = random_state
    elif isinstance(random_state, int):
        rng = np.random.RandomState(random_state)

    cliffords_1q = (cirq.X, cirq.Y, cirq.Z, cirq.H)
    cliffords_2q = (cirq.CZ, cirq.CNOT, cirq.SWAP)
    moments = []
    for _ in range(n_moments):
        moment_ops = []
        n_layer_qubits = rng.binomial(n_qubits, op_density)
        layer_qubits = list(
            rng.choice(qubits, size=n_layer_qubits, replace=False))
        while any(layer_qubits):
            sampler = cliffords_1q
            if len(layer_qubits) > 1:
                sampler += cliffords_2q
            gate = rng.choice(sampler)
            gate_qubits = [layer_qubits.pop() for _ in range(gate.num_qubits())]
            moment_ops.append(gate(*gate_qubits))
        moments += moment_ops
    return cirq.Circuit(*moments)
