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
#include "../qsim/lib/circuit_noisy.h"
#include "../qsim/lib/fuser.h"
#include "../qsim/lib/gates_cirq.h"
#include "absl/container/flat_hash_map.h"
#include "cirq_google/api/v2/program.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"

namespace tfq {

enum GateParamNames { kExponent = 0, kPhaseExponent, kTheta, kPhi };

struct GateMetaData {
  // Struct for additional metadata about a specific gate.
  // Any new parsing features should add needed information
  // to this struct and then proceed to process the data
  // outside of the parsing code.

  // symbol name strings found in gate placeholders (if any).
  std::vector<std::string> symbol_values;

  // ids of placeholders
  std::vector<GateParamNames> placeholder_names;

  // index of gate in qsim circuit.
  int index;

  // list of params from protobuf used when constructing
  // this gate. Note: this vector will be different from
  // the *params* vector in qsim gates.
  // this vector will exclude: time, qubit locs etc.
  std::vector<float> gate_params;

  // set only if gate is Single qubit Eigen gate.
  std::function<qsim::Cirq::GateCirq<float>(unsigned int, unsigned int, float,
                                            float)>
      create_f1;

  // set only if gate is Two qubit Eigen gate.
  std::function<qsim::Cirq::GateCirq<float>(unsigned int, unsigned int,
                                            unsigned int, float, float)>
      create_f2;
};

// parse a serialized Cirq program into a qsim representation.
// ingests a Cirq Circuit proto and produces a resolved qsim Circuit,
// as well as a fused circuit.
tensorflow::Status QsimCircuitFromProgram(
    const cirq::google::api::v2::Program& program,
    const absl::flat_hash_map<std::string, std::pair<int, float>>& param_map,
    const int num_qubits, qsim::Circuit<qsim::Cirq::GateCirq<float>>* circuit,
    std::vector<qsim::GateFused<qsim::Cirq::GateCirq<float>>>* fused_circuit,
    std::vector<GateMetaData>* metdata = nullptr);

// parse a serialized Cirq program into a qsim representation.
// ingests a Cirq Circuit proto and produces a resolved Noisy qsim Circuit.
// If add_tmeasures is true then terminal measurements are added on all
// qubits.
// Note: no metadata or fused circuits are produced as the qsim api for
// 	noisy simulation appears to take care of a lot of this for us.
tensorflow::Status NoisyQsimCircuitFromProgram(
    const cirq::google::api::v2::Program& program,
    const absl::flat_hash_map<std::string, std::pair<int, float>>& param_map,
    const int num_qubits, const bool add_tmeasures,
    qsim::NoisyCircuit<qsim::Cirq::GateCirq<float>>* ncircuit);

// parse a serialized pauliTerm from a larger cirq.Paulisum proto
// into a qsim Circuit and fused circuit.
tensorflow::Status QsimCircuitFromPauliTerm(
    const tfq::proto::PauliTerm& term, const int num_qubits,
    qsim::Circuit<qsim::Cirq::GateCirq<float>>* circuit,
    std::vector<qsim::GateFused<qsim::Cirq::GateCirq<float>>>* fused_circuit);

// parse a serialized pauliTerm from a larger cirq.Paulisum proto
// into a qsim Circuit and fused circuit that represents the transformation
// to the z basis.
tensorflow::Status QsimZBasisCircuitFromPauliTerm(
    const tfq::proto::PauliTerm& term, const int num_qubits,
    qsim::Circuit<qsim::Cirq::GateCirq<float>>* circuit,
    std::vector<qsim::GateFused<qsim::Cirq::GateCirq<float>>>* fused_circuit);

}  // namespace tfq

#endif  // TFQ_CORE_SRC_CIRCUIT_PARSER_QSIM_H_
