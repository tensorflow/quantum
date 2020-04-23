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
"""Input checks common to circuit execution layers."""
import numpy as np
import sympy
import tensorflow as tf

import cirq
from tensorflow_quantum.python import util


def expand_circuits(inputs, symbol_names=None, symbol_values=None):
    """Function for consistently expanding input circuits.
        
    Args:
        inputs: a single `cirq.Circuit`, a Python `list` of
            `cirq.Circuit`s, or a pre-converted `tf.Tensor` of
            `cirq.Circuit`s.
        symbol_names: a Python `list` of `str` or `sympy.Symbols`,
            or a pre-converted `tf.Tensor` of type `str`.
        symbol_values: a Python `list` of floating point values,
            or `np.ndarray` or pre-converted `tf.Tensor` of floats.

    Returns:
        inputs: `tf.Tensor` of strings with shape [batch_size] containing
            the string representations of the circuits to be executed.
        symbol_names: `tf.Tensor` of strings with shape [n_params], which
            is used to specify the order in which the values in
            `symbol_values` should be placed inside of the circuits in
            `programs`.
        symbol_values: `tf.Tensor` of real numbers with shape
            [batch_size, n_params] specifying parameter values to resolve
            into the circuits specified by programs, following the ordering
            dictated by `symbol_names`.
    """
    # inputs is the circuit(s).
    symbols_empty = False
    if symbol_names is None:
        symbol_names = []
    if symbol_values is None:
        symbols_empty = True
        symbol_values = [[]]

    # Ingest and promote symbol_names.
    if isinstance(symbol_names, (list, tuple, np.ndarray)):
        if symbol_names and not all(
                [isinstance(x, (str, sympy.Symbol)) for x in symbol_names]):
            raise TypeError("Each element in symbol_names"
                            " must be a string or sympy.Symbol.")
        symbol_names = [str(s) for s in symbol_names]
        if not len(symbol_names) == len(list(set(symbol_names))):
            raise ValueError("All elements of symbol_names must be unique.")
        symbol_names = tf.convert_to_tensor(symbol_names,
                                            dtype=tf.dtypes.string)
    elif tf.is_tensor(symbol_names):
        if not symbol_names.dtype == tf.dtypes.string:
            raise TypeError("symbol_names tensor must have dtype string.")
        if not symbol_names.shape[0] == len(list(set(symbol_names.numpy()))):
            raise ValueError("All elements of symbol_names must be unique.")   
    else:
        raise TypeError("symbol_names must be list-like.")    
        
    # Ingest and promote symbol_values.
    if isinstance(symbol_values, (list, tuple, np.ndarray)):
        symbol_values = tf.convert_to_tensor(symbol_values,
                                             dtype=tf.dtypes.float32)
    elif tf.is_tensor(symbol_values):
        if not symbol_values.dtype == tf.dtypes.float32:
            raise TypeError("symbol_values tensor must have dtype float32.")
    else:
        raise TypeError("symbol_values must be list-like.")

    # Ingest and promote circuit.
    symbol_batch_dim = tf.gather(tf.shape(symbol_values), 0)
    if isinstance(inputs, cirq.Circuit):
        # process single circuit.
        inputs = tf.tile(util.convert_to_tensor([inputs]),
                         [symbol_batch_dim])
    elif isinstance(inputs, (list, tuple, np.ndarray)):
        # process list of circuits.
        inputs = util.convert_to_tensor(inputs)
    elif tf.is_tensor(inputs):
        if not inputs.dtype == tf.dtypes.string:
            raise TypeError("inputs tensor must contain serialized circuits.")
    else:
        raise TypeError("inputs must be a single Circuit, a list of "
                        "Circuits, or a tensor of serialized Circuits.")

    if symbols_empty:
        # No symbol_values were provided. so we must tile up the
        # symbol values so that symbol_values = [[]] * number of circuits
        # provided.
        circuit_batch_dim = tf.gather(tf.shape(inputs), 0)
        symbol_values = tf.tile(symbol_values,
                                tf.stack([circuit_batch_dim, 1]))

    return inputs, symbol_names, symbol_values