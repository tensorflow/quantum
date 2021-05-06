"""A module for all the circuit and readout building blocks.

Does minimal error checks, but is not completely comprehensive.
"""
import cirq

def ghz_standard(qubits, rotations):
    """Make a GHZ-like state with arbitrary phase.

    Prepare a|0000> + b|1111>. Where a and b depend on the values
    in the rotations list.

    Args:
        qubits: Python `lst` of `cirq.GridQubit`s
        rotations: Python `lst` indicating the X half rotations, Y half
            rotations and Z half rotations.
    """
    if len(rotations) != 3:
        raise ValueError("Number of needed rotations is 3.")
    return cirq.Circuit(
        cirq.Z(qubits[0])**rotations[0],
        cirq.X(qubits[0])**rotations[1],
        cirq.Z(qubits[0])**rotations[2]) + cirq.Circuit(
            cirq.CNOT(q0, q1) for q0, q1 in zip(qubits, qubits[1:]))

def ghz_cz(qubits, rotations):
    """Make a GHZ-like state with arbitrary phase using CZ gates.
    For the purposes of the noise experiment, we don't apply Z phase
    corrections, since the point is to match the generator and data
    gate parameters to know that there's high state overlap.

    Args:
        qubits: Python `lst` of `cirq.GridQubit`s
        rotations: Python `lst` indicating the X half rotations, Y half
            rotations and Z half rotations.
    """
    if len(rotations) != 3:
        raise ValueError("Number of needed rotations is 3.")
    
    u = [cirq.Z(qubits[0])**rotations[0],
        cirq.X(qubits[0])**rotations[1],
        cirq.Z(qubits[0])**rotations[2]]
    for q0, q1 in zip(qubits, qubits[1:]):
        u.extend([cirq.Y(q1)**0.5, cirq.X(q1), cirq.CZ(q0, q1),
                  cirq.Y(q1)**0.5, cirq.X(q1)])
    return cirq.Circuit(u)


def variational_swap_textbook(qubits_a, qubits_b, ancilla, rotations):
    """Make a variational swap test circuit using ancilla method.

    Args:
        qubits_a: Python `lst` of `cirq.GridQubit`s indicating subsystem A's
            qubits.
        qubits_b: Python `lst` of `cirq.GridQubit`s indicating subsystem B's
            qubits.
        rotations: Python `lst` of shape [n_qubits, 2] containing Y,X rotation
            parameters for the swap test.
    """
    c = cirq.Circuit()
    for i, (q0, q1) in enumerate(zip(qubits_a, qubits_b)):
        c += cirq.H(ancilla)
        c += cirq.decompose([
            cirq.CCNOT(ancilla, q0, q1),
            cirq.CCNOT(ancilla, q1, q0),
            cirq.CCNOT(ancilla, q0, q1)
        ])
        c += cirq.Circuit(
            cirq.Y(ancilla)**rotations[i][0],
            cirq.X(ancilla)**rotations[i][1])
    return c


def ancilla_free_variational_swap(qubits_a, qubits_b, rotations):
    """Make a variational swap test circuit.

    Args:
        qubits_a: Python `lst` of `cirq.GridQubit`s indicating subsystem A's
            qubits.
        qubits_b: Python `lst` of `cirq.GridQubit`s indicating subsystem B's
            qubits.
        rotations: Python `lst` of shape [n_qubits, 2] containing Y,X rotation
            parameters for the swap test.
    """
    if len(rotations) != len(qubits_a) or any(len(x) != 2 for x in rotations):
        raise ValueError("rotations must be shape [len(qubits_a), 2]")

    if len(qubits_a) != len(qubits_b):
        raise ValueError("unequal system sizes.")

    c = cirq.Circuit(cirq.CNOT(q0, q1) for q0, q1 in zip(qubits_a, qubits_b))
    # This becomes the perfect swap test with rotations=[[0.5, 1.0]] * len(qubits_a)
    for i, q in enumerate(qubits_a):
        c += cirq.Y(q)**rotations[i][0]
        c += cirq.X(q)**rotations[i][1]

    return c

def cz_swap(qubits_a, qubits_b, rotations):
    """Make a variational swap test circuit.

    Args:
        qubits_a: Python `lst` of `cirq.GridQubit`s indicating subsystem A's
            qubits.
        qubits_b: Python `lst` of `cirq.GridQubit`s indicating subsystem B's
            qubits.
        rotations: Python `lst` of shape [n_qubits, 2] containing Z rotation
            parameters for the swap test.
    """
    if len(rotations) != len(qubits_a) or any(len(x) != 2 for x in rotations):
        raise ValueError("rotations must be shape [len(qubits_a), 2]")

    if len(qubits_a) != len(qubits_b):
        raise ValueError("unequal system sizes.")
    
    u = []
    for i in range(len(qubits_a)):
        q0 = qubits_a[i]
        q1 = qubits_b[i]
        u.extend([cirq.Y(q1)**0.5, cirq.X(q1), cirq.CZ(q0, q1), cirq.Z(q0)**rotations[i][0], cirq.Z(q1)**rotations[i][1], cirq.Y(q1)**0.5, cirq.X(q1)])
    
    # expanded Hadamard: H = X Y^(1/2)
    for i, q in enumerate(qubits_a):
        u.append(cirq.Y(q)**0.5)
        u.append(cirq.X(q)**1.0)

    return cirq.Circuit(u)


def variational_swap_iswap(qubits_a, qubits_b, rotations):
    """Make a variational swap test circuit using ISWAP gates.

    Args:
        qubits_a: Python `lst` of `cirq.GridQubit`s indicating subsystem A's
            qubits.
        qubits_b: Python `lst` of `cirq.GridQubit`s indicating subsystem B's
            qubits.
        rotations: Python `lst` of shape [n_qubits, 2] containing Y,X rotation
            parameters for the swap test.
    """
    circuit = ancilla_free_variational_swap(qubits_a, qubits_b, rotations)
    cirq.google.optimizers.ConvertToSqrtIswapGates().optimize_circuit(circuit)
    return circuit


def _countSetBits(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


def _one_proj(a):
    return 0.5 * (1 - cirq.Z(a))


def swap_readout_op(qubits_a, qubits_b):
    """Readout operation for variational swap test.

    Computes the bitwise and of matched qubits from qubits_a and qubits_b.

    When the states have perfect overlap the expectation of this op will be -1
    when these states are orthogonal the expectation of this op will be 1.

    Args:
        qubits_a: Python `lst` of `cirq.GridQubit`s. The qubits system A act on
        qubits_b: Python `lst` of `cirq.GridQubit`s. The qubits system B act on
    """
    if len(qubits_a) != len(qubits_b):
        raise ValueError("unequal system sizes.")

    ret_op = 0
    for i in range(1 << len(qubits_a)):
        if _countSetBits(i) % 2 == 0:
            tmp_op = 1
            for j, ch in enumerate(bin(i)[2:].zfill(len(qubits_a))):
                intermediate = _one_proj(qubits_a[j]) * _one_proj(qubits_b[j])
                if ch == '0':
                    intermediate = 1 - intermediate
                tmp_op *= intermediate
            ret_op += tmp_op

    return 1.0 - (ret_op * 2 - 1)


def swap_textbook_op(ancilla):
    """Readout operation for the ancilla based swap test.

    Args:
        ancilla: `cirq.GridQubit` that the swap test uses.
    """
    return 1.0 - (cirq.Z(ancilla))