"""A module for building the GAN models."""
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import circuits


def build_ghz_no_ancilla_generator(generator_qubits,
                               data_qubits,
                               generator_symbols,
                               lr,
                               generator_initialization,
                               noise_model,
                               backend=None,
                               use_sampled=False,
                               circuit_type='standard'):
    """Build a generator tf.keras.Model using standard circuits.

    Args:
        generator_qubits: Python `lst` of `cirq.GridQubit`s indicating the
            qubits that the generator should use.
        data_qubits: Python `lst` of `cirq.GridQubit`s indicating the qubits
            that the data will arrive on.
        generator_symbols: Python `lst` of numbers or `sympy.Symbol`s
            to use in the ansatze used for the generator.
        lr: Python `float` the learning rate of the model.
        backend: Python object for the backend type to use when running quantum
            circuits.
        generator_initialization: `np.ndarray` of initial values to place
            inside of the generator symbols in the tensorflow managed
            variables.
        use_sampled: Python `bool` indicating whether or not to use analytical
            expectation or sample based expectation calculation.
        circuit_type: Python `str` indicating whether to use standard gateset
            or cz gateset.
    """

    ghz_circuit = circuits.ghz_standard
    if circuit_type == 'cz':
        ghz_circuit = circuits.ghz_cz
    
    return build_generic_no_ancilla_generator(ghz_circuit,
                               generator_qubits,
                               data_qubits,
                               generator_symbols,
                               lr,
                               generator_initialization,
                               noise_model,
                               backend=backend,
                               use_sampled=use_sampled,
                               circuit_type=circuit_type)

def build_generic_no_ancilla_generator(generator_circuit,
                               generator_qubits,
                               data_qubits,
                               generator_symbols,
                               lr,
                               generator_initialization,
                               noise_model,
                               backend=None,
                               use_sampled=False,
                               circuit_type='standard',
                               regularization=0.000001):
    """Build a generator tf.keras.Model using standard circuits.

    Args:
        generator_circuit: Python function that will take `(generator_qubits, 
            generator_symbols)` to create the circuit.
        generator_qubits: Python `lst` of `cirq.GridQubit`s indicating the
            qubits that the generator should use.
        data_qubits: Python `lst` of `cirq.GridQubit`s indicating the qubits
            that the data will arrive on.
        generator_symbols: Python `lst` of numbers or `sympy.Symbol`s
            to use in the ansatze used for the generator.
        lr: Python `float` the learning rate of the model.
        backend: Python object for the backend type to use when running quantum
            circuits.
        generator_initialization: `np.ndarray` of initial values to place
            inside of the generator symbols in the tensorflow managed
            variables.
        use_sampled: Python `bool` indicating whether or not to use analytical
            expectation or sample based expectation calculation.
        circuit_type: Python `str` indicating whether to use standard gateset
            or cz gateset.
        regularization: Python `float` added as margin to an orthogonal swap test.
    """

    # Input for the circuits that generate the quantum data from the source.
    signal_input = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)

    # Input for the swaptest circuits. These will have the variables from the
    # discriminator resolved into them.
    swap_test_input = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)

    data_and_generated = tfq.layers.AddCircuit()(signal_input,
                                                 append=generator_circuit(
                                                     generator_qubits,
                                                     generator_symbols).
                                                     with_noise(noise_model))

    # Append the variational swap test on to the data on data_qubits
    # and the "generated" data on generator_qubits.
    full_swaptest = tfq.layers.AddCircuit()(data_and_generated,
                                            append=swap_test_input)

    expectation_output = None
    if not use_sampled:
        expectation_output = tfq.layers.Expectation(backend=backend)(
            full_swaptest,
            symbol_names=generator_symbols,
            operators=circuits.swap_readout_op(generator_qubits, data_qubits),
            initializer=tf.constant_initializer(generator_initialization))

    else:
        expectation_output = tfq.layers.SampledExpectation(backend=backend)(
            full_swaptest,
            symbol_names=generator_symbols,
            operators=circuits.swap_readout_op(generator_qubits, data_qubits),
            initializer=tf.constant_initializer(generator_initialization),
            repetitions=10000)

    expectation_output = tf.add(expectation_output, tf.constant(regularization))
    log_output = tf.math.log(expectation_output)

    # Input is true data on data qubits, and swap_test_input for both qubits.
    qgan_g_model = tf.keras.Model(inputs=[signal_input, swap_test_input],
                                  outputs=[expectation_output, log_output])

    optimizerg = tf.keras.optimizers.Adam(learning_rate=lr)
    lossg = lambda x, y: tf.reduce_mean(y)
    qgan_g_model.compile(optimizer=optimizerg, loss=lossg, loss_weights=[0,1])

    return qgan_g_model


def build_ghz_ancilla_generator(generator_qubits,
                                data_qubits,
                                ancilla,
                                generator_symbols,
                                lr,
                                generator_initialization,
                                noise_model,
                                backend=None,
                                use_sampled=False):
    """Build a generator tf.keras.Model using ancilla based circuits.

    Args:
        generator_qubits: Python `lst` of `cirq.GridQubit`s indicating the
            qubits that the generator should use.
        data_qubits: Python `lst` of `cirq.GridQubit`s indicating the qubits
            that the data will arrive on.
        ancilla: `cirq.GridQubit` ancilla qubit used for swap test.
        generator_symbols: Python `lst` of numbers or `sympy.Symbol`s
            to use in the ansatze used for the generator.
        lr: Python `float` the learning rate of the model.
        backend: Python object for the backend type to use when running quantum
            circuits.
        generator_initialization: `np.ndarray` of initial values to place
            inside of the generator symbols in the tensorflow managed
            variables.
        use_sampled: Python `bool` indicating whether or not to use analytical
            expectation or sample based expectation calculation.
    """

    # Input for the circuits that generate the quantum data from the source.
    signal_input = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)

    # Input for the swaptest circuits. These will have the variables from the
    # discriminator resolved into them.
    swap_test_input = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)

    data_and_generated = tfq.layers.AddCircuit()(signal_input,
                                                 append=circuits.ghz_standard(
                                                     generator_qubits,
                                                     generator_symbols).
                                                     with_noise(noise_model))

    # Append the variational swap test on to the data on data_qubits
    # and the "generated" data on generator_qubits.
    full_swaptest = tfq.layers.AddCircuit()(data_and_generated,
                                            append=swap_test_input)

    expectation_output = None
    if not use_sampled:
        expectation_output = tfq.layers.Expectation(backend=backend)(
            full_swaptest,
            symbol_names=generator_symbols,
            operators=circuits.swap_textbook_op(ancilla),
            initializer=tf.constant_initializer(generator_initialization))

    else:
        expectation_output = tfq.layers.SampledExpectation(backend=backend)(
            full_swaptest,
            symbol_names=generator_symbols,
            operators=circuits.swap_textbook_op(ancilla),
            initializer=tf.constant_initializer(generator_initialization))

    log_output = tf.math.log(expectation_output)

    # Input is true data on data qubits, and swap_test_input for both qubits.
    qgan_g_model = tf.keras.Model(inputs=[signal_input, swap_test_input],
                                  outputs=[expectation_output, log_output])

    optimizerg = tf.keras.optimizers.Adam(learning_rate=lr)
    lossg = lambda x, y: tf.reduce_mean(y)
    qgan_g_model.compile(optimizer=optimizerg, loss=lossg, loss_weights=[0,1])

    return qgan_g_model


class SharedVar(tf.keras.layers.Layer):
    """A custom tf.keras.layers.Layer used for sharing variables."""
    def __init__(self, symbol_names, operators, init_vals, backend,
                 use_sampled):
        """Custom keras layer used to share tf.Variables between several
        tfq.layers.Expectation."""
        super(SharedVar, self).__init__()
        self.init_vals = init_vals
        self.symbol_names = symbol_names
        self.operators = operators
        self.use_sampled = use_sampled
        self.backend = backend

    def build(self, input_shape):
        # Build a tf.Variable that is the shape of the number of symbols.
        self.w = self.add_weight(shape=(len(self.symbol_names),),
                                 initializer=tf.constant_initializer(
                                     self.init_vals))

    def call(self, inputs):
        # inputs[0] = circuit tensor
        # inputs[1] = circuit tensor
        # Their expectations are evaluated with shared variables between them
        n_datapoints = tf.gather(tf.shape(inputs[0]), 0)
        values = tf.tile(tf.expand_dims(self.w, 0), [n_datapoints, 1])
        if not self.use_sampled:
            return tfq.layers.Expectation(backend=self.backend)(
                inputs[0],
                symbol_names=self.symbol_names,
                operators=self.operators,
                symbol_values=values), tfq.layers.Expectation(
                    backend=self.backend)(inputs[1],
                                          symbol_names=self.symbol_names,
                                          operators=self.operators,
                                          symbol_values=values)
        else:
            return tfq.layers.SampledExpectation(backend=self.backend)(
                inputs[0],
                symbol_names=self.symbol_names,
                operators=self.operators,
                symbol_values=values,
                repetitions=10000), tfq.layers.SampledExpectation(
                    backend=self.backend)(inputs[1],
                                          symbol_names=self.symbol_names,
                                          operators=self.operators,
                                          symbol_values=values,
                                          repetitions=10000)


def build_no_ancilla_discriminator(generator_qubits,
                                   data_qubits,
                                   discriminator_symbols,
                                   lr,
                                   discriminator_initialization,
                                   noise_model,
                                   backend=None,
                                   use_sampled=False,
                                   circuit_type='standard',
                                   regularization=0.000001):
    """Build a discriminator model.

    Args:
        generator_qubits: Python `lst` of `cirq.GridQubit`s indicating the
            qubits that the generator should use.
        data_qubits: Python `lst` of `cirq.GridQubit`s indicating the qubits
            that the data will arrive on.
        discriminator_symbols: Python `lst` of numbers or `sympy.Symbol`s
            to use in the ansatze used for the discriminator.
        lr: Python `float` the learning rate of the model.
        discriminator_initialization: `np.ndarray` of symbols to place
            inside of the discriminator symbols in the tensorflow managed
            variables.
        backend: Python object for the backend type to use when running quantum
            circuits.
        use_sampled: Python `bool` indicating whether or not to use analytical
            expectation or sample based expectation calculation.
        circuit_type: Python `str` indicating whether to use standard gateset
            or cz gateset.
        regularization: Python `float` added as margin to an orthogonal swap test.
    """

    swaptest_circuit = circuits.cz_swap
    if circuit_type == 'standard':
        swaptest_circuit = circuits.ancilla_free_variational_swap

    # True data on data_qubits.
    signal_input_d = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)

    # Generator data on generator_qubits.
    load_generator_data_d = tf.keras.layers.Input(shape=(),
                                                  dtype=tf.dtypes.string)

    # True data on generator_qubits.
    load_true_data_d = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)

    # Swap circuit with input.
    swap_test_input_d = tfq.layers.AddCircuit()(
        signal_input_d,
        append=swaptest_circuit(data_qubits, generator_qubits,
                                np.array(discriminator_symbols).reshape(-1, 2)).
                                with_noise(noise_model))
    

    # Swap test between the true data and generator.
    swaptest_d = tfq.layers.AddCircuit()(load_generator_data_d,
                                         append=swap_test_input_d)

    # Swap test between the true data and itself. Useful for how close to the
    # "true" swap test we are over time as we train.
    swapontruedata = tfq.layers.AddCircuit()(load_true_data_d,
                                             append=swap_test_input_d)

    tmp = SharedVar(discriminator_symbols,
                    circuits.swap_readout_op(generator_qubits, data_qubits),
                    discriminator_initialization, backend, use_sampled)
    expectation_output_d, expectation_output2 = tmp(
        [swaptest_d, swapontruedata])

    expectation_output_d = tf.add(expectation_output_d, tf.constant(regularization))
    log_discrim_dist = tf.math.log(tf.keras.backend.flatten(expectation_output_d))
    log_true_dist = tf.math.log(tf.keras.backend.flatten(expectation_output2))


    final_output = -log_discrim_dist# + log_true_dist

    qgan_d_model = tf.keras.Model(
        inputs=[signal_input_d, load_generator_data_d, load_true_data_d],
        outputs=[expectation_output_d, expectation_output2, final_output])

    optimizerd = tf.keras.optimizers.Adam(learning_rate=lr)

    # Difference between "generator vs true data" and "true vs true (given
    # we many not be doing a perfect swap test yet)"
    lossd = lambda x, y: -tf.reduce_mean(y)
    qgan_d_model.compile(optimizer=optimizerd, loss=lossd, loss_weights=[0,0,1])

    return qgan_d_model


def build_ancilla_discriminator(generator_qubits,
                                data_qubits,
                                ancilla,
                                discriminator_symbols,
                                lr,
                                discriminator_initialization,
                                noise_model,
                                backend=None,
                                use_sampled=False):
    """Build a discriminator model.

    Args:
        generator_qubits: Python `lst` of `cirq.GridQubit`s indicating the
            qubits that the generator should use.
        data_qubits: Python `lst` of `cirq.GridQubit`s indicating the qubits
            that the data will arrive on.
        discriminator_symbols: Python `lst` of numbers or `sympy.Symbol`s
            to use in the ansatze used for the discriminator.
        lr: Python `float` the learning rate of the model.
        discriminator_initialization: `np.ndarray` of symbols to place
            inside of the discriminator symbols in the tensorflow managed
            variables.
        backend: Python object for the backend type to use when running quantum
            circuits.
        use_sampled: Python `bool` indicating whether or not to use analytical
            expectation or sample based expectation calculation.
        circuit_type: Python `str` indicating whether to use standard gateset
            or cz gateset.
    """

    # True data on data_qubits.
    signal_input_d = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)

    # Generator data on generator_qubits.
    load_generator_data_d = tf.keras.layers.Input(shape=(),
                                                  dtype=tf.dtypes.string)

    # True data on generator_qubits.
    load_true_data_d = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)

    swap_test_input_d = tfq.layers.AddCircuit()(
        signal_input_d,
        append=circuits.variational_swap_textbook(
            data_qubits, generator_qubits, ancilla,
            np.array(discriminator_symbols).reshape((-1, 2))).
            with_noise(noise_model))

    # Swap test between the true data and generator.
    swaptest_d = tfq.layers.AddCircuit()(load_generator_data_d,
                                         append=swap_test_input_d)

    # Swap test between the true data and itself. Useful for how close to the
    # "true" swap test we are over time as we train.
    swapontruedata = tfq.layers.AddCircuit()(load_true_data_d,
                                             append=swap_test_input_d)

    tmp = SharedVar(discriminator_symbols,
                    circuits.swap_textbook_op(ancilla),
                    discriminator_initialization, backend, use_sampled)

    expectation_output_d, expectation_output2 = tmp(
        [swaptest_d, swapontruedata])

    qgan_d_model = tf.keras.Model(
        inputs=[signal_input_d, load_generator_data_d, load_true_data_d],
        outputs=[expectation_output_d, expectation_output2])

    log_discrim_dist = tf.math.log(tf.keras.backend.flatten(expectation_output_d))
    log_true_dist = tf.math.log(tf.keras.backend.flatten(expectation_output2))


    final_output = -log_discrim_dist# + log_true_dist

    qgan_d_model = tf.keras.Model(
        inputs=[signal_input_d, load_generator_data_d, load_true_data_d],
        outputs=[expectation_output_d, expectation_output2, final_output])

    optimizerd = tf.keras.optimizers.Adam(learning_rate=lr)

    # Difference between "generator vs true data" and "true vs true (given
    # we many not be doing a perfect swap test yet)"
    lossd = lambda x, y: -tf.reduce_mean(y)
    qgan_d_model.compile(optimizer=optimizerd, loss=lossd, loss_weights=[0,0,1])

    return qgan_d_model