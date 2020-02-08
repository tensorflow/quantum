/* Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TFQ_CORE_SRC_CIRCUIT_PARSER_H_
#define TFQ_CORE_SRC_CIRCUIT_PARSER_H_

#include <algorithm>
#include <string>
#include <vector>

#include "cirq/google/api/v2/program.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/src/circuit.h"

namespace tfq {

// parse a serialized Cirq program into our internal representation
tensorflow::Status CircuitFromProgram(
    const cirq::google::api::v2::Program& program, const int num_qubits,
    Circuit* circuit);

// build the circuit taking the computational basis to the measurement basis
tensorflow::Status CircuitFromPauliTerm(const tfq::proto::PauliTerm& term,
                                        const int num_qubits, Circuit* circuit);

}  // namespace tfq

#endif  // TFQ_CORE_SRC_CIRCUIT_PARSER_H_
