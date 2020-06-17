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

#include "tensorflow_quantum/core/src/circuit_parser_qsim.h"

#include <google/protobuf/text_format.h>

#include <string>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/gates_cirq.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "cirq/google/api/v2/program.pb.h"
#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status.h"

namespace tfq {
namespace {

typedef absl::flat_hash_map<std::string, std::pair<int, float>> SymbolMap;
typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;

// using ::cirq::google::api::v2::Program;
// using ::qsim::Cirq::GateCirq;

TEST(QsimCircuitParserTest, qsimthing) {
  ::cirq::google::api::v2::Program program_proto;
  ::cirq::google::api::v2::Circuit* circuit_proto =
      program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  ::cirq::google::api::v2::Moment* moments_proto = circuit_proto->add_moments();

  // Add CNOT gate.
  ::cirq::google::api::v2::Operation* operations_proto =
      moments_proto->add_operations();
  ::cirq::google::api::v2::Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("CNP");

  // Add args to gate.
  ::cirq::google::api::v2::Arg global_shift_arg;
  ::cirq::google::api::v2::ArgValue* global_shift_arg_value =
      global_shift_arg.mutable_arg_value();
  global_shift_arg_value->set_float_value(0.0);
  ::cirq::google::api::v2::Arg exponent_arg;
  ::cirq::google::api::v2::ArgValue* exponent_arg_value =
      exponent_arg.mutable_arg_value();
  exponent_arg_value->set_float_value(1.0);
  ::cirq::google::api::v2::Arg exponent_scalar_arg;
  ::cirq::google::api::v2::ArgValue* exponent_scalar_arg_value =
      exponent_scalar_arg.mutable_arg_value();
  exponent_scalar_arg_value->set_float_value(1.0);

  google::protobuf::Map<std::string, ::cirq::google::api::v2::Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["global_shift"] = global_shift_arg;
  (*args_proto)["exponent"] = exponent_arg;
  (*args_proto)["exponent_scalar"] = exponent_scalar_arg;

  // Set the qubits.
  ::cirq::google::api::v2::Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");
  qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("1");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap empty_map;

  ASSERT_EQ(QsimCircuitFromProgram(program_proto, empty_map, 2, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status::OK());
}

}  // namespace
}  // namespace tfq
