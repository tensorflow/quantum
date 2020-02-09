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

#include "tensorflow_quantum/core/src/circuit_parser.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "cirq/google/api/v2/program.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/src/circuit.h"
#include "tensorflow_quantum/core/src/gates_def.h"

namespace tfq {
namespace {

using ::cirq::google::api::v2::Moment;
using ::cirq::google::api::v2::Operation;
using ::cirq::google::api::v2::Program;
using ::cirq::google::api::v2::Qubit;
using ::tensorflow::Status;

// Adds the operation as a Gate in the circuit. The index is the moment number.
// TODO(pmassey): Remove duplciate code between this and ../src/moment
Status ParseOperation(const Operation& op, const int num_qubits,
                      const int index, Gate* gate) {
  std::vector<unsigned int> locations;
  unsigned int location;
  for (const Qubit& qubit : op.qubits()) {
    if (!absl::SimpleAtoi(qubit.id(), &location)) {
      return Status(tensorflow::error::INVALID_ARGUMENT,
                    "Could not parse Qubit id: " + qubit.ShortDebugString());
    }
    locations.push_back(num_qubits - location - 1);
  }

  // Control and target are swapped relative to cirq convention
  std::reverse(locations.begin(), locations.end());

  absl::flat_hash_map<std::string, float> arg_map;
  for (const auto& pair : op.args()) {
    arg_map[pair.first] = pair.second.arg_value().float_value();
  }

  const std::string& gate_name = op.gate().id();
  return InitGate(gate_name, index, locations, arg_map, gate);
}

}  // namespace

Status CircuitFromProgram(const Program& program, const int num_qubits,
                          Circuit* circuit) {
  circuit->num_qubits = num_qubits;

  const cirq::google::api::v2::Circuit& cirq_circuit = program.circuit();
  if (cirq_circuit.scheduling_strategy() !=
      cirq::google::api::v2::Circuit::MOMENT_BY_MOMENT) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Circuit must be moment by moment.");
  }

  int i = 0;
  for (const Moment& moment : cirq_circuit.moments()) {
    for (const Operation& op : moment.operations()) {
      Gate gate;
      Status status = ParseOperation(op, num_qubits, i, &gate);
      if (!status.ok()) {
        return status;
      }

      // Assert that q0 is always less than q1.
      // Note that gate construction must handle this, so this error should
      // only be thrown if a gate implementation is incorrect.
      if (gate.num_qubits == 2 && gate.qubits[0] > gate.qubits[1]) {
        return Status(
            tensorflow::error::INVALID_ARGUMENT,
            "Gate has q0 > q1 for operation: " + op.ShortDebugString());
      }

      circuit->gates.push_back(gate);
    }

    i++;
  }

  // TODO(zaqqwerty): identities hack added to collect orphan gates
  if (num_qubits > 1) {
    I2GateBuilder builder;
    std::vector<unsigned int> locations;
    absl::flat_hash_map<std::string, float> arg_map;
    Gate gate;
    for (int w = 0; w < num_qubits - 1; w += 2) {
      locations.clear();
      locations.push_back(w);
      locations.push_back(w + 1);
      builder.Build(i, locations, arg_map, &gate);
      circuit->gates.push_back(gate);
    }
    i++;
    if (num_qubits % 2 == 1) {
      locations.clear();
      locations.push_back(num_qubits - 2);
      locations.push_back(num_qubits - 1);
      builder.Build(i, locations, arg_map, &gate);
      circuit->gates.push_back(gate);
    }
  }

  return Status::OK();
}

Status CircuitFromPauliTerm(const tfq::proto::PauliTerm& term,
                            const int num_qubits, Circuit* circuit) {
  Program measurement_program;
  measurement_program.mutable_circuit()->set_scheduling_strategy(
      cirq::google::api::v2::Circuit::MOMENT_BY_MOMENT);
  Moment* term_moment = measurement_program.mutable_circuit()->add_moments();
  for (const tfq::proto::PauliQubitPair& pair : term.paulis()) {
    Operation* new_op = term_moment->add_operations();

    // create corresponding eigen gate op.
    new_op->add_qubits()->set_id(pair.qubit_id());
    new_op->mutable_gate()->set_id(pair.pauli_type() + "P");
    (*new_op->mutable_args())["exponent"].mutable_arg_value()->set_float_value(
        1.0);
    (*new_op->mutable_args())["global_shift"]
        .mutable_arg_value()
        ->set_float_value(0.0);
    (*new_op->mutable_args())["exponent_scalar"]
        .mutable_arg_value()
        ->set_float_value(1.0);
  }

  return CircuitFromProgram(measurement_program, num_qubits, circuit);
}

}  // namespace tfq
