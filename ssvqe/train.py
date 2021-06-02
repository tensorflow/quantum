import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
from pickle import load
from math import prod

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def layer(circuit, qubits, parameters):
    for i in range(len(qubits)):
        circuit += cirq.ry(parameters[3*i]).on(qubits[i])
        circuit += cirq.rz(parameters[3*i+1]).on(qubits[i])
        circuit += cirq.ry(parameters[3*i+2]).on(qubits[i])
    for i in range(len(qubits)-1):
        circuit += cirq.CNOT(qubits[i], qubits[i+1])
    circuit += cirq.CNOT(qubits[-1], qubits[0])
    return circuit

def ansatz(circuit, qubits, layers, parameters):
    for i in range(layers):
        params = parameters[3*i*len(qubits):3*(i+1)*len(qubits)]
        circuit = layer(circuit, qubits, params)
    return circuit

def exp_val(qubits, hamiltonian):
    return prod([op(qubits[i]) for i, op in enumerate(hamiltonian) if hamiltonian[i] != 0])

class VQE(tf.keras.layers.Layer):
    def __init__(self, circuits, ops):
        super(VQE, self).__init__()
        self.layers = [tfq.layers.ControlledPQC(circuits[i], ops[i], differentiator=tfq.differentiators.Adjoint()) for i in range(len(circuits))]

    def call(self, inputs):
        return sum([self.layers[i]([inputs[0], inputs[1]]) for i in range(len(self.layers))])

class SSVQE(tf.keras.layers.Layer):
    def __init__(self, num_weights, circuits, ops, k, const):
        super(SSVQE, self).__init__()
        self.theta = tf.Variable(np.random.uniform(0, 2 * np.pi, (1, num_weights)), dtype=tf.float32)
        self.hamiltonians = []
        self.k = k
        self.const = const
        for i in range(k):
            self.hamiltonians.append(VQE(circuits[i], ops[i]))

    def call(self, inputs):
        total = 0
        energies = []
        for i in range(self.k):
            c = self.hamiltonians[i]([inputs, self.theta]) + self.const
            energies.append(c)
            if i == 0:
               total += c
            else:
                total += ((0.9 - i * 0.1) * c)
        return total, energies

def make_ssvqe(num_qubits, layers, coeffs, hamilton, k):
    qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
    num_params = layers * 3 * num_qubits
    params = sympy.symbols('vqe0:%d'%num_params)

    ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    circuits = []
    op = []
    const = 0

    for j in range(k):
        circuits.append([])
        op.append([])
        for i in range(len(hamilton)):
            if len(hamilton[i]) == 0:
                const = coeffs[i]
                continue
            readout_ops = coeffs[i] * exp_val(qubits, hamilton[i])
            op[j].append(readout_ops)
            if j == 0:
                circuits[j].append(ansatz(cirq.Circuit(), qubits, layers, params))
            else:
                circuits[j].append(ansatz(cirq.Circuit(cirq.X(qubits[j % num_qubits])), qubits, layers, params))

    v = SSVQE(num_params, circuits, op, k, const)(ins)
    vqe_model = tf.keras.models.Model(inputs=ins, outputs=v[0])
    energies = SSVQE(num_params, circuits, op, k, const)
    return vqe_model, energies, const


def train_ssvqe(ssvqe, opt, tol=5e-6, patience=10):
    ssvqe_model = ssvqe[0]
    energies = ssvqe[1]
    prev_loss = 100
    counter = 0
    inputs = tfq.convert_to_tensor([cirq.Circuit()])
    while True:
        with tf.GradientTape() as tape:
            loss = ssvqe_model(inputs)
        grads = tape.gradient(loss, ssvqe_model.trainable_variables)
        opt.apply_gradients(zip(grads, ssvqe_model.trainable_variables))
        loss = loss.numpy()[0][0]
        if abs(loss - prev_loss) < tol:
            counter += 1
        if counter > patience:
            break
        prev_loss = loss

    energies.theta = ssvqe_model.trainable_variables[0]
    energies = [i.numpy()[0][0] for i in energies(inputs)[1]]
    return energies[0], energies[1]


diatomic_bond_length = 0.2
interval = 0.1
max_bond_length = 2.0   
ground_energies_real = []
ground_energies_vqe = []
excited_energies_real = []
excited_energies_vqe = []
bond_lengths = []
k = 2

# VQE Hyperparameters
layers = 4
n_qubits = 4
optimizer = tf.keras.optimizers.Adam(lr=0.1)

with open("real", "rb") as r_file:
    real = load(r_file)

step = 0
while diatomic_bond_length <= max_bond_length:
    print(diatomic_bond_length, max_bond_length)
    eigs = real[step]
    ham_name = "mol_hamiltonians_" + str(step)
    coef_name = "coef_hamiltonians_" + str(step)
    with open(ham_name, "rb") as ham_file:
        hamiltonians = load(ham_file)
    with open(coef_name, "rb") as coeff_file:
        coefficients = load(coeff_file)
    ssvqe = make_ssvqe(n_qubits, layers, coefficients, hamiltonians, k)
    ground, excited = train_ssvqe(ssvqe, optimizer)
    print(ground - eigs[0], excited - eigs[1])
    ground_energies_vqe.append(ground)
    excited_energies_vqe.append(excited)
    ground_energies_real.append(eigs[0])
    excited_energies_real.append(eigs[1])
    bond_lengths.append(diatomic_bond_length)
    diatomic_bond_length += interval
    step += 1


plt.scatter(bond_lengths, ground_energies_real, label='Ground State', marker='o', facecolors="None", edgecolor='blue')
plt.scatter(bond_lengths, excited_energies_real, label='First Excited State', marker='o', facecolors="None", edgecolor='red')
plt.plot(bond_lengths, ground_energies_vqe, label='VQE Predicted Ground State', color='blue')
plt.plot(bond_lengths, excited_energies_vqe, label='VQE Predicted First Excited State', color='red')
plt.xlabel("Interatomic Distance (Angstroms)")
plt.ylabel("Energy (Hartree)")
plt.legend()
plt.show()
plt.savefig('oftest2')

