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

#ifndef TFQ_CORE_SRC_CIRCUIT_PARSER_QSIM_H_
#define TFQ_CORE_SRC_CIRCUIT_PARSER_QSIM_H_

#include <string>
#include <vector>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/fuser.h"
#include "../qsim/lib/gates_cirq.h"
#include "absl/container/flat_hash_map.h"
#include "cirq/google/api/v2/program.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tfq {

// parse a serialized Cirq program into a qsim representation.
// ingests a Cirq Circuit proto and produces a resolved qsim Circuit,
// as well as a fused circuit.
tensorflow::Status QsimCircuitFromProgram(
    const cirq::google::api::v2::Program& program,
    const absl::flat_hash_map<std::string, std::pair<int, float>>& param_map,
    const int num_qubits, qsim::Circuit<qsim::Cirq::GateCirq<float>>* circuit,
    std::vector<qsim::GateFused<qsim::Cirq::GateCirq<float>>>* fused_circuit);

}  // namespace tfq

#endif  // TFQ_CORE_SRC_CIRCUIT_PARSER_QSIM_H_
