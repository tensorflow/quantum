from scipy.sparse import linalg
import openfermion
from openfermionpyscf import generate_molecular_hamiltonian
import numpy as np
from pickle import dump

diatomic_bond_length = 0.2
interval = 0.1
max_bond_length = 2.0   
basis = 'sto-3g'
multiplicity = 1
charge = 0
ground_energies_real = []
ground_energies_vqe = []
excited_energies_real = []
excited_energies_vqe = []
bond_lengths = []
k = 2
step = 0

full = []

while diatomic_bond_length <= max_bond_length:
    print(diatomic_bond_length, max_bond_length)
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., diatomic_bond_length))]
    molecular_hamiltonian = generate_molecular_hamiltonian(geometry, basis, multiplicity, charge)
    n_qubits = openfermion.count_qubits(molecular_hamiltonian)
    jw_operator = openfermion.transforms.jordan_wigner(molecular_hamiltonian)
    hamiltonian_jw_sparse = openfermion.get_sparse_operator(jw_operator)
    eigs, _ = linalg.eigsh(hamiltonian_jw_sparse, k=k, which='SA')
    operators = openfermion.transforms.qubit_operator_to_pauli_sum(jw_operator)
    coefficients = []
    hamiltonians = []
    for op in operators:
        sub_hamiltonain = list(op.items())
        coefficients.append(np.real(op.coefficient))
        hamilton = [0] * n_qubits
        if len(sub_hamiltonain) == 0:
            hamiltonians.append([])
            continue
        for i in sub_hamiltonain:
            hamilton[i[0].x] = i[1]
        hamiltonians.append(hamilton)

    ham_name = "mol_hamiltonians_" + str(step)
    coef_name = "coef_hamiltonians_" + str(step)
    with open(ham_name, "wb") as fp:
        dump(hamiltonians, fp)
    with open(coef_name, "wb") as fp:
        dump(coefficients, fp)

    diatomic_bond_length += interval
    step += 1
    full.append(list(eigs))

with open("real", "wb") as fp:
    dump(full, fp)

