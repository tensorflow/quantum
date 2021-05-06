"""A module for carrying out model training."""
import cirq
import gin
import sympy
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq

import matplotlib.pyplot as plt

import circuits
import models
from analyze import find_best_params

def get_circuit_maker(c_type):
    """Get appropriate circuit maker for a given circuit type."""
    circuit_maker = None
    if c_type in ['standard', 'ancilla']:
        circuit_maker = circuits.ghz_standard
    elif c_type == 'cz':
        circuit_maker = circuits.ghz_cz
    elif c_type == 'iswap':
        circuit_maker = circuits.ghz_iswap
    elif c_type == 'exp0':
        circuit_maker = circuits.exp0_ansatz
    elif c_type == 'exp1':
        circuit_maker = circuits.exp1_ansatz
    return circuit_maker

def get_data_maker(c_type):
    """Get appropriate dataset maker for a given circuit type."""
    data_maker = None
    if c_type in ['standard', 'ancilla']:
        data_maker = circuits.ghz_standard
    elif c_type == 'cz':
        data_maker = circuits.ghz_cz
    elif c_type == 'iswap':
        data_maker = circuits.ghz_iswap
    elif c_type == 'exp0':
        data_maker = circuits.exp0_truth
    elif c_type == 'exp1':
        data_maker = circuits.exp1_truth
    return data_maker

def num_gen_parameters(c_type, n_qubits):
    """Get number of generator model parameters for a circuit type."""
    if c_type in ['cz', 'standard', 'ancilla', 'iswap']:
        return 3
    elif c_type == 'exp0':
        return n_qubits - 1
    elif c_type == 'exp1':
        return 2*n_qubits + 1
    return None

def num_true_parameters(c_type, n_qubits):
    """Get number of true data circuit parameters for a circuit type."""
    if c_type == 'exp0':
        return n_qubits - 1
    elif c_type == 'exp1':
        return n_qubits - 1
    else:
        return num_gen_parameters(c_type, n_qubits)

def num_disc_parameters(c_type, n_qubits):
    """Get number of discriminator model parameters for a circuit type."""
    return 2*n_qubits

def get_rand_state(c_type, n_qubits, data_noise):
    """Get number of data preparation circuit parameters for a circuit type."""
    return np.random.uniform(-data_noise, data_noise, num_true_parameters(c_type, n_qubits))


# add controlled phase and Z phase errors after each CZ gate
# CZ phase error is fully random
# Z phase error is always the same for a given qubit index
class CZNoiseModel(cirq.NoiseModel):
    def __init__(self, qubits, mean, stdev, seed=0):
        self.mean = mean
        self.stdev = stdev

        np.random.seed(seed)
        single_errors = {}
        for q in qubits:
            single_errors[q] = np.random.normal(self.mean[1], self.stdev[1])
        self.single_errors = single_errors

    def noisy_operation(self, op):
        if isinstance(op.gate, cirq.ops.CZPowGate):
            return [
                op,
                cirq.ops.CZPowGate(
                    exponent=np.random.normal(self.mean[0], self.stdev[0])
                )(*op.qubits),
                cirq.ops.ZPowGate(exponent=self.single_errors[op.qubits[0]])(
                    op.qubits[0]
                ),
                cirq.ops.ZPowGate(exponent=self.single_errors[op.qubits[1]])(
                    op.qubits[1]
                ),
            ]

        return op


def quantum_data_overlap(qubits, params_a, params_b, c_type):
    """Compute overlap of quantum data circuits with params_a and params_b."""
    sim = cirq.Simulator()
    circuit_maker = get_circuit_maker(c_type)
    data_maker = get_data_maker(c_type)
    circuit_a = circuit_maker(qubits, params_a)
    circuit_b = data_maker(qubits, params_b)
    res_a = cirq.final_state_vector(circuit_a)
    res_b = cirq.final_state_vector(circuit_b)
    overlap = np.abs(np.vdot(res_a, res_b))
    return overlap


def generate_data(
    data_qubits,
    generator_qubits,
    target_quantum_data,
    data_noise,
    noise_model,
    n_points,
    c_type,
):
    """Generate n_points data on data_qubits with generator_qubits linked for later copying."""
    data_maker = get_data_maker(c_type)

    target_circuits = []
    target_real_data_circuit = []

    rand_states = []
    for i in range(n_points):
        rand_states.append(get_rand_state(c_type, len(data_qubits), data_noise))
    for rand_state in rand_states:
        rand_circuit = data_maker(data_qubits, rand_state + target_quantum_data)
        rand_circuit_true_data_on_generator_qubit = data_maker(
            generator_qubits, rand_state + target_quantum_data
        )

        c_data = rand_circuit.with_noise(noise_model)
        c_gen = rand_circuit_true_data_on_generator_qubit.with_noise(noise_model)

        target_circuits.append(c_data)
        target_real_data_circuit.append(c_gen)
    target_circuits = tfq.convert_to_tensor(target_circuits)
    target_real_data_circuit = tfq.convert_to_tensor(target_real_data_circuit)

    return target_circuits, target_real_data_circuit


# def run_experiment(d_learn, g_learn, d_epoch, g_epoch, batchsize, data_noise, n_data,
#                    n_episodes, n_qubits, c_type, backend, use_sampled,
#                    log_interval, target_quantum_data, generator_initialization,
#                    discriminator_initialization, use_perfect_swap, gate_error_mean,
#                    gate_error_stdev, seed, save):

def run_experiment(
    d_learn,
    g_learn,
    d_epoch,
    g_epoch,
    batchsize,
    data_noise,
    n_data,
    n_episodes,
    n_qubits,
    c_type,
    backend,
    use_sampled,
    log_interval,
    target_quantum_data,
    generator_initialization,
    discriminator_initialization,
    use_perfect_swap,
    gate_error_mean,
    gate_error_stdev,
    seed,
    save
):
    """Run a QGAN experiment.

    Args:
        d_learn: Python `float` discriminator learning rate.
        g_learn: Python `float` generator learning rate.
        d_epoch: Python `int` number of discriminator iterations per episode.
        g_epoch: Python `int` number of generator iterations per episode.
        batchsize: Python `int` number of entries to use in a batch.
        data_noise: Python `float` bounds on noise in real data preparation.
        n_data: Python `int` number of total datapoints to generate.
        n_episodes: Python `int` number of total QGAN training episodes.
        n_qubits: Python `int` number of qubits to use for each susbsystem.
        c_type: Python `str` indicating 'standard' or 'ancilla'.
        backend: None or `cirq.SimulatesFinalState` or `cirq.Sampler`.
        use_samples: Python `bool` whether or not analytical or sampled exp.
        log_interval: Python `int` log every log_interval episodes.
        target_quantum_data: Python object. True target state.
        generator_initialization: `np.ndarray` initial gen guess.
        discriminator_initialization: `np.ndarray` initial disc guess.
        use_perfect_swap: `bool` whether or not to train discriminator.
        gate_error_mean: mean of angle error on 2-qubit gates (normal distribution).
        gate_error_stdev: standard deviation of angle error on 2-qubit gates.
        seed: seed of run for noise model and training.
        save: filename suffix if to be saved, None if not to be saved.
    """
    circuit_maker = get_circuit_maker(c_type)
    if generator_initialization is None:
        generator_initialization = np.zeros(num_gen_parameters(c_type, n_qubits))
    if use_perfect_swap or (discriminator_initialization is None):
        # initialize to perfect swap test
        if c_type != "standard":
            discriminator_initialization = np.array([[0.0, 0.0]] * n_qubits)
        else:
            discriminator_initialization = np.array([[0.5, 1.0]] * n_qubits)

    # Create data and generator qubits.
    data_qubits = [cirq.GridQubit(1, k + 4) for k in range(n_qubits)]
    generator_qubits = [cirq.GridQubit(2, k + 4) for k in range(n_qubits)]
    ancilla = cirq.GridQubit(1, 5)  # potentially unused.
    all_qubits = data_qubits + generator_qubits

    # Noise on single-qubit gates
    if (gate_error_mean is None) or (gate_error_stdev is None):
        noise_model = None
    else:
        noise_model = CZNoiseModel(
            all_qubits, gate_error_mean, gate_error_stdev, seed=seed
        )

    # Generator and Discriminator symbols.
    discriminator_parameters = []
    generator_parameters = []
    for j in range(num_disc_parameters(c_type, n_qubits)):
        discriminator_parameters.append(sympy.Symbol("Discrimx{!r}".format(j)))
    for j in range(num_gen_parameters(c_type, n_qubits)):
        generator_parameters.append(sympy.Symbol("Genx{!r}".format(j)))
    target_circuits, target_real_data_circuit = generate_data(
        data_qubits,
        generator_qubits,
        target_quantum_data,
        data_noise,
        noise_model,
        n_data,
        c_type,
    )

    qgan_d_model = None
    if c_type == "ancilla":
        qgan_d_model = models.build_ancilla_discriminator(
            generator_qubits,
            data_qubits,
            ancilla,
            discriminator_parameters,
            d_learn,
            discriminator_initialization,
            noise_model,
            None,
            use_sampled,
        )
    else:
        qgan_d_model = models.build_no_ancilla_discriminator(
            generator_qubits,
            data_qubits,
            discriminator_parameters,
            d_learn,
            discriminator_initialization,
            noise_model,
            None,
            use_sampled,
            c_type,
        )
    qgan_g_model = None
    if c_type == "ancilla":
        qgan_g_model = models.build_ghz_ancilla_generator(
            generator_qubits,
            data_qubits,
            ancilla,
            generator_parameters,
            g_learn,
            generator_initialization,
            noise_model,
            None,
            use_sampled,
        )
    else:
        qgan_g_model = models.build_ghz_no_ancilla_generator(
            generator_qubits,
            data_qubits,
            generator_parameters,
            g_learn,
            generator_initialization,
            noise_model,
            None,
            use_sampled,
            c_type,
        )

    # Tracking info
    d_loss = []
    g_loss = []
    overlap_record = []
    param_history = []

    # First train generator with frozen perfect discriminator.
    repeats = 1
    if not use_perfect_swap:
        repeats = 2
        n_episodes = n_episodes // 2

    for r in range(repeats):
        if r == 0:  # use perfect swap on first iteration regardless of experiment
            use_perfect_swap = True
        elif (
            r == 1
        ):  # if we're doing a second iteration, then adversarial phase is required
            use_perfect_swap = False
        # Begin training adversarially.
        for k in range(1, n_episodes + 1):
            t = time.time()
            if k != 0:
                generator_initialization = qgan_g_model.trainable_variables[0].numpy()
                if any(np.isnan(generator_initialization)):
                    print("NAN")
                    break

            overlap_record.append(
                quantum_data_overlap(
                    data_qubits, generator_initialization, target_quantum_data, c_type
                )
            )
            param_history.append(
                [
                    qgan_g_model.trainable_variables[0].numpy(),
                    qgan_d_model.trainable_variables[0].numpy(),
                ]
            )

            if not use_perfect_swap:
                ###### prepare discriminator network input
                gen_circuit = circuit_maker(generator_qubits, generator_initialization)
                gen_circuit = gen_circuit.with_noise(noise_model)
                load_generator_circuit = tf.tile(
                    tfq.convert_to_tensor([gen_circuit]), tf.constant([n_data])
                )

                historyd = qgan_d_model.fit(
                    x=[
                        target_circuits,
                        load_generator_circuit,
                        target_real_data_circuit,
                    ],
                    y=[
                        tf.zeros_like(target_circuits, dtype=tf.float32),
                        tf.zeros_like(target_circuits, dtype=tf.float32),
                        tf.zeros_like(target_circuits, dtype=tf.float32),
                    ],
                    epochs=d_epoch,
                    batch_size=batchsize,
                    verbose=0,
                )

                d_loss.append(historyd.history["loss"])

                ###### prepare generator network input
                discriminator_initialization = (
                    qgan_d_model.trainable_variables[0].numpy().reshape((-1, 2))
                )

            swap_test_circuit = None
            if c_type == "ancilla":
                swap_test_circuit = circuits.variational_swap_textbook(
                    data_qubits, generator_qubits, ancilla, discriminator_initialization
                )
            else:  # cz GHZ, exp, etc.
                swap_test_circuit = circuits.cz_swap(
                    data_qubits, generator_qubits, discriminator_initialization
                )

            swap_test_circuit = swap_test_circuit.with_noise(noise_model)
            swap_test_circuit = tf.tile(
                tfq.convert_to_tensor([swap_test_circuit]), tf.constant([n_data])
            )

            history = qgan_g_model.fit(
                x=[target_circuits, swap_test_circuit],
                y=[
                    tf.zeros_like(target_circuits, dtype=tf.float32),
                    tf.zeros_like(target_circuits, dtype=tf.float32),
                ],
                epochs=g_epoch,
                batch_size=batchsize,
                verbose=0,
            )

            g_loss.append(history.history["loss"])

            if k % log_interval == 0:
                print(f"Step = {k}. Time={time.time() - t}")
                print(f"Step = {k}. Overlap={overlap_record[-1]}")
                print(f"Step = {k}. g_loss={g_loss[-1]}")
                if not use_perfect_swap:
                    print(f"Step = {k}. d_loss={d_loss[-1]}")
                print(
                    f"Step = {k}. gen_params={qgan_g_model.trainable_variables[0].numpy()}"
                )
                print(
                    f"Step = {k}. discrim_params={qgan_d_model.trainable_variables[0].numpy()}"
                )

                print("-" * 50)

    return g_loss, d_loss, overlap_record, param_history







@gin.configurable
def run_and_save(d_learn, g_learn, d_epoch, g_epoch, batchsize, data_noise, n_data,
                 n_episodes, n_qubits, c_type, backend, use_sampled,
                 log_interval, target_quantum_data, generator_initialization,
                 discriminator_initialization, use_perfect_swap, gate_error_mean,
                 gate_error_stdev, averages, save):
    """Run and plot an experiment."""
    if backend == None:
        backend = None
    elif backend == 'engine_mcgee':
        backend = testsamplerxmon_mcgee
    elif backend == 'engine_rainbow':
        backend = testsamplerxmon_rainbow
    else:
        raise ValueError('Invalid backend.')
    
    if use_perfect_swap == 'both':
        use_perfect_swaps = [False, True]
    else:
        use_perfect_swaps = [use_perfect_swap]
    
    # all history of all runs
    all_g_loss = []
    all_d_loss = []
    all_overlap_record = []
    all_g_weights = []
    all_d_weights = []
    
    # relevant statistics
    all_chosen_params = []
    all_chosen_overlaps = []
    
    for j in range(len(use_perfect_swaps)):
        all_g_loss.append([])
        all_d_loss.append([])
        all_overlap_record.append([])
        all_g_weights.append([])
        all_d_weights.append([])
        all_chosen_params.append([])
        all_chosen_overlaps.append([])
    
    for i in range(averages):
        for j in range(len(use_perfect_swaps)):
            g_loss, d_loss, overlap_record, weights = run_experiment(
                d_learn, g_learn, d_epoch, g_epoch, batchsize, data_noise, n_data, n_episodes,
                n_qubits, c_type, backend, use_sampled, log_interval,
                target_quantum_data, generator_initialization, discriminator_initialization,
                use_perfect_swaps[j], gate_error_mean, gate_error_stdev, i, save)

            weights = np.array(weights, dtype=object)
            g_weights = weights[:, 0]
            d_weights = weights[:, 1]

            _, chosen_params, chosen_overlap = find_best_params(overlap_record, g_weights, g_loss)
            all_chosen_params[j].append(chosen_params)
            all_chosen_overlaps[j].append(chosen_overlap)

            all_g_loss[j].append(np.array(g_loss).flatten())
            all_d_loss[j].append(np.array(d_loss).flatten())
            all_overlap_record[j].append(overlap_record)
            all_g_weights[j].append(g_weights)
            all_d_weights[j].append(d_weights)
    
    if save is not None:
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        for j in range(len(use_perfect_swaps)):
            out = [np.array(all_overlap_record[j]), np.array(all_g_weights[j]),
                    np.array(all_d_weights[j]), np.array(all_g_loss[j]), np.array(all_d_loss[j])]
            if use_perfect_swaps[j]:
                fname = 'out-all-perfect_swap-' + save + '-' + date_time + '.npz'
            else:
                fname = 'out-all-adversarial_swap-' + save + '-' + date_time + '.npz'
            np.savez(fname, overlap_record=out[0],
                     g_weights=out[1],
                     d_weights=out[2],
                     g_loss=out[3],
                     d_loss=out[4])

    print(all_chosen_overlaps)
    print("Done!")
    return np.mean(all_chosen_overlaps)
