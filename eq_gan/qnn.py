import tensorflow as tf
import tensorflow_quantum as tfq
from datetime import datetime

import cirq
import sympy
import numpy as np
import collections
import itertools

from skopt import gp_minimize
from skopt.space.space import Real

tune = False
backend=None

# pre-trained weights from EQ-GAN on exactly the same training set
# QRAM is trained from 60 examples (half of the size = 120)
all_weights = [[1.3459893, 1.0012823, 0.94282967], [4.7395287, 0.96802247]]

# helper function
def unpackbits(x, num_bits):
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits).reshape([1, num_bits])
    return np.flip((x & mask).astype(bool).astype(int).reshape(xshape + [num_bits]), axis=1)

# create 2-peak dataset
def create_data(seed, n=4, dataset_size=100):
    np.random.seed(seed)
    # sample data from Gaussian
    data0_raw = np.random.normal(2**(n-1), scale=2, size=dataset_size)
    bins = np.arange(2**n + 1).astype(np.float64)
    bins[-1] = np.inf
    counts0, _ = np.histogram(data0_raw, bins=bins)
    data0 = np.clip(np.floor(data0_raw), 0, 2**n - 1)

    data1_raw = np.random.normal(2**(n-2), scale=1, size=dataset_size)
    counts1, _ = np.histogram(data1_raw, bins=bins)
    data1 = np.clip(np.floor(data1_raw), 0, 2**n - 1)
    
    return data0, data1


# QNN for generator: double exponential peaks
def build_qnn(qubits, model_type):
    n = len(qubits)
    u = []
    angles = []
    if model_type == 0:
        center = 0
        j = 0
        for i in range(n):
            if i == center:
                u.extend([cirq.Y(qubits[i])**0.5, cirq.X(qubits[i])])
            else:
                theta = sympy.Symbol('t' + str(i))
                angles.append(theta)
                u.append(cirq.ry(2*theta).on(qubits[i]))
                j += 1
        for i in range(n):
            if i != center:
                u.extend([cirq.Y(qubits[i])**0.5, cirq.X(qubits[i]),
                                            cirq.CZ(qubits[center], qubits[i]),
                                            cirq.Y(qubits[i])**0.5, cirq.X(qubits[i])])
        circuit = cirq.Circuit(u)
    elif model_type == 1:
        j = 0
        center = 1
        u.append(cirq.I.on(qubits[0]))
        for i in range(1, n):
            if i == center:
                u.extend([cirq.Y(qubits[i])**0.5, cirq.X(qubits[i])])
            else:
                theta = sympy.Symbol('t' + str(i))
                angles.append(theta)
                u.append(cirq.ry(2*theta).on(qubits[i]))
                j += 1
        for i in range(1, n):
            if i != center:
                u.extend([cirq.Y(qubits[i])**0.5, cirq.X(qubits[i]),
                                            cirq.CZ(qubits[center], qubits[i]),
                                            cirq.Y(qubits[i])**0.5, cirq.X(qubits[i])])
        circuit = cirq.Circuit(u)
    return circuit, angles
    
# get qubits for a rainbow chip
def get_exp_qubits(class_type=-1):
    if class_type == 0:
        return [cirq.GridQubit(2, 4), cirq.GridQubit(1, 4), cirq.GridQubit(2, 3), cirq.GridQubit(2, 5), cirq.GridQubit(3, 4)]
    elif class_type == 1:
        return [cirq.GridQubit(1, 4), cirq.GridQubit(2, 4), cirq.GridQubit(2, 3), cirq.GridQubit(2, 5), cirq.GridQubit(3, 4)]
    else:
        return [cirq.GridQubit(3, 4), cirq.GridQubit(1, 4), cirq.GridQubit(2, 3), cirq.GridQubit(2, 5), cirq.GridQubit(2, 4)]
    # return cirq.GridQubit.rect(1, n)
    
# do a swap gate with CZ between q0 and q1
def compiled_swap(q0, q1):
    u = []
    u.extend([cirq.X(q0)**0.5])
    u.extend([cirq.Z(q1)**-0.5, cirq.X(q1)**0.5, cirq.Z(q1)**0.5])
    u.append(cirq.CZ(q0, q1))
    u.extend([cirq.Z(q0)**-1, cirq.X(q0)**0.5, cirq.Z(q0)**1])
    u.extend([cirq.Z(q1)**-1.5, cirq.X(q1)**0.5, cirq.Z(q1)**1.5])
    u.append(cirq.CZ(q0, q1))
    u.extend([cirq.X(q0)**0.5])
    u.extend([cirq.Z(q1)**-0.5, cirq.X(q1)**0.5, cirq.Z(q1)**0.5])
    u.append(cirq.CZ(q0, q1))
    u.extend([cirq.Z(q0)**-0.5])
    u.extend([cirq.Z(q1)**0.5])
    return cirq.Circuit(u)
    
# get a learned circuit for a given dataset
def get_model(class_type):
    qubits = get_exp_qubits(class_type)
    qnn, symbols = build_qnn(qubits[:-1], class_type)
    resolver = {}
    for i in range(len(symbols)):
        resolver[symbols[i]] = all_weights[class_type][i]
    resolved_qnn = cirq.resolve_parameters(qnn, resolver)

    all_qubits = get_exp_qubits()
    resolved_qnn += compiled_swap(all_qubits[0], all_qubits[-1])
    return resolved_qnn
    
    
# create circuits from dataset (for sampling)
size = 120
n = 4 # only works for n=4 since we hard-coded qubits
data0, data1 = create_data(0, n=n, dataset_size=size)
all_data = np.array([unpackbits(data0.astype(np.int64), n), unpackbits(data1.astype(np.int64), n)])

def convert_to_circuit(image):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = get_exp_qubits()
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        circuit.append(cirq.X(qubits[i])**value)
    return circuit
    
x_circ = [convert_to_circuit(x) for x in all_data[0]] + [convert_to_circuit(x) for x in all_data[1]]
y = np.array([0]*len(all_data[0]) + [1]*len(all_data[1]))

x_train_tfcirc = tfq.convert_to_tensor(x_circ[:size//2])
x_test_tfcirc = tfq.convert_to_tensor(x_circ[size//2:])
y_train = y[:size//2]
y_test = y[size//2:]


# define the QNN classifier
class ClassifierCircuitLayerBuilder():
    def __init__(self, data_qubits, readouts):
        self.data_qubits = data_qubits
        self.readouts = readouts
    
    def add_layer(self, circuit, prefix):
        for j, readout in enumerate(self.readouts):
            for i, qubit in enumerate(self.data_qubits):
                symbol = sympy.Symbol(prefix + '-' + str(j) + '-' + str(i))
                # if 'xx' in prefix:
                #       circuit += cirq.Circuit(cirq.XX(qubit, readout)**symbol)
                # elif 'zz' in prefix:
                #       circuit += cirq.Circuit(cirq.ZZ(qubit, readout)**symbol)
                if 'xx' in prefix:
                    u = []
                    u.extend([cirq.Z(readout)**-0.5, cirq.X(readout)**0.5, cirq.Z(readout)**0.5])
                    u.append(cirq.CZ(qubit, readout))
                    u.extend([cirq.Z(qubit)**-1, cirq.X(qubit)**symbol, cirq.Z(qubit)**1])
                    u.append(cirq.CZ(qubit, readout))
                    u.extend([cirq.Z(readout)**0.5, cirq.X(readout)**0.5, cirq.Z(readout)**-0.5])
                    circuit += cirq.Circuit(u)
                elif 'zz' in prefix: # not quite a ZZ gate
                    u = []
                    u.extend([cirq.Z(qubit)**-0.5, cirq.X(qubit)**0.5, cirq.Z(qubit)**0.5])
                    u.append(cirq.CZ(qubit, readout))
                    u.extend([cirq.Z(qubit)**-1, cirq.X(qubit)**symbol, cirq.Z(qubit)**1])
                    u.append(cirq.CZ(qubit, readout))
                    u.extend([cirq.Z(qubit)**0.5, cirq.X(qubit)**0.5, cirq.Z(qubit)**-0.5])
                    circuit += cirq.Circuit(u)
        
def build_quantum_classifier(n_readouts=1):
    """Create a QNN model circuit and readout operation to go along with it."""
    readouts = []
    qubits = get_exp_qubits()
    for i in range(n_readouts):
        readouts.append(qubits[-1])
    circuit = cirq.Circuit()
    data_qubits = qubits[:-1]
    
    # Prepare the readout qubit.
    circuit.append(cirq.X.on_each(readouts))
    circuit.append((cirq.Y**0.5).on_each(readouts))
    circuit.append(cirq.X.on_each(readouts))
    
    builder = ClassifierCircuitLayerBuilder(data_qubits, readouts)

    # Then add layers (experiment by adding more).
#     builder.add_layer(circuit, "xx1")
    builder.add_layer(circuit, "zz1")

    # Finally, prepare the readout qubit.
    
    circuit.append((cirq.Y**0.5).on_each(readouts))
    circuit.append(cirq.X.on_each(readouts))

    total = cirq.Z(readouts[0])
    for readout in readouts[1:]:
        total += cirq.Z(readout)
    return circuit, total/len(readouts)
    
# create the QNN classifier
N_READOUTS = 1
model_circuit, model_readout = build_quantum_classifier(N_READOUTS)

y_train_hinge = 2.0*y_train-1.0
y_test_hinge = 2.0*y_test-1.0

def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)
    
EPOCHS = 1

# train non-superposition QNN classifier
def train_sample_qnn(averages=5, learning_rate=0.001, save=True):
    sample_acc_data = []
    for i in range(averages):
        model = tf.keras.Sequential([
                # The input is the data-circuit, encoded as a tf.string
                tf.keras.layers.Input(shape=(), dtype=tf.string),
                # The PQC layer returns the expected value of the readout gate, range [-1,1].
                tfq.layers.PQC(model_circuit, model_readout, repetitions=10000, backend=backend),
        ])
        model.compile(
                loss=tf.keras.losses.Hinge(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=[hinge_accuracy])
    
        qnn_history_sample = model.fit(
                    x_train_tfcirc, y_train_hinge,
                    batch_size=1,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(x_test_tfcirc, y_test_hinge))
    
        qnn_results_sample = model.evaluate(x_test_tfcirc, y_test)
        sample_weights = model.get_weights()[0]
        sample_acc_data.append(qnn_results_sample[1])
        
        if save:
            now = datetime.now()
            date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
            fname = 'qnn_out/qnn-sample-'
            np.save(fname + 'hinge_accuracy-' + date_time + '.npy', qnn_history_sample.history['hinge_accuracy'])
            np.save(fname + 'loss-' + date_time + '.npy', qnn_history_sample.history['loss'])
            np.save(fname + 'val_hinge_accuracy-' + date_time + '.npy', qnn_history_sample.history['val_hinge_accuracy'])
            np.save(fname + 'val_loss-' + date_time + '.npy', qnn_history_sample.history['val_loss'])
            np.save(fname + 'all_test_accuracies-' + date_time + '.npy', sample_acc_data)
    return sample_acc_data
    
# train superposition QNN classifier
def train_superpos_qnn(averages=5, learning_rate=0.001, save=True):
        gen_circuit_class_0 = get_model(0)
        gen_circuit_class_1 = get_model(1)
        superposition_acc_data = []
        for i in range(averages):
            model = tf.keras.Sequential([
                    # The input is the data-circuit, encoded as a tf.string
                    tf.keras.layers.Input(shape=(), dtype=tf.string),
                    # The PQC layer returns the expected value of the readout gate, range [-1,1].
                    tfq.layers.PQC(model_circuit, model_readout, repetitions=10000, backend=backend),
            ])
            model.compile(
                    loss=tf.keras.losses.Hinge(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    metrics=[hinge_accuracy])

            x_superposition = tfq.convert_to_tensor([gen_circuit_class_0, gen_circuit_class_1])
            y_superposition = np.array([-1, 1])

            qnn_history_superposition = model.fit(
                        x_superposition, y_superposition,
                        batch_size=1,
                        epochs=EPOCHS*len(x_train_tfcirc)//2,
                        verbose=1,
                        validation_data=(x_superposition, y_superposition))

            qnn_results_superposition = model.evaluate(x_test_tfcirc, y_test)
            superposition_weights = model.get_weights()[0]
            superposition_acc_data.append(qnn_results_superposition[1])
            
            if save:
                now = datetime.now()
                date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
                fname = './qnn_out/qnn-superpos-'
                np.save(fname + 'hinge_accuracy-' + date_time + '.npy', qnn_history_superposition.history['hinge_accuracy'])
                np.save(fname + 'loss-' + date_time + '.npy', qnn_history_superposition.history['loss'])
                np.save(fname + 'val_hinge_accuracy-' + date_time + '.npy', qnn_history_superposition.history['val_hinge_accuracy'])
                np.save(fname + 'val_loss-' + date_time + '.npy', qnn_history_superposition.history['val_loss'])
                np.save(fname + 'all_test_accuracies-' + date_time + '.npy', superposition_acc_data)
        return superposition_acc_data


# for xx1 and zz1
# lr_tunes = {'superposition': 10**-1.79, 'sample': 10**-4.00}

# for just zz1
lr_tunes = {'superposition': 10**-1.83, 'sample': 10**-3.93}

# run the QNNs
if tune:
    averages = 10
    lr_range = [Real(-4, -1)]

    def opt_helper_superpos(lr):
        return -np.mean(train_superpos_qnn(averages=averages, learning_rate=10**lr[0], save=False))
    def opt_helper_sample(lr):
        return -np.mean(train_sample_qnn(averages=averages, learning_rate=10**lr[0], save=False))

    res_sup = gp_minimize(opt_helper_superpos, lr_range, n_calls=50)
    res_sam = gp_minimize(opt_helper_sample, lr_range, n_calls=50)
    print("Superposition: x*=%.2f f(x*)=%.2f" % (res_sup.x[0], res_sup.fun))
    print("Sample: x*=%.2f f(x*)=%.2f" % (res_sam.x[0], res_sam.fun))
else:

    averages = 50
        
    superpos_qnn_data = train_superpos_qnn(averages=averages, learning_rate=lr_tunes['superposition'])
    print('QNN superposition accuracy (mean):', np.mean(superpos_qnn_data))
    print('QNN superposition accuracy (stdev):', np.std(superpos_qnn_data)/np.sqrt(averages))

    sample_qnn_data = train_sample_qnn(averages=averages, learning_rate=lr_tunes['sample'])
    print('QNN sample accuracy (mean):', np.mean(sample_qnn_data))
    print('QNN sample accuracy (stdev):', np.std(sample_qnn_data)/np.sqrt(averages))