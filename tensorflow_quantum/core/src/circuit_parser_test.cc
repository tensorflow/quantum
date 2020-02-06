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

#include <google/protobuf/text_format.h>

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "cirq/api/google/v2/program.pb.h"
#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/src/circuit.h"
#include "tensorflow_quantum/core/src/circuit_parser.h"
#include "tensorflow_quantum/core/src/gates_def.h"

namespace tfq {
namespace {

using ::cirq::api::google::v2::Program;

TEST(CircuitParserTest, CircuitFromProgramInvalidSchedule) {
  Program program_proto;
  ::cirq::api::google::v2::Circuit* circuit_proto =
      program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(
      circuit_proto->SCHEDULING_STRATEGY_UNSPECIFIED);
  Circuit test_circuit;
  ASSERT_EQ(CircuitFromProgram(program_proto, 0, &test_circuit),
            tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                               "Circuit must be moment by moment."));
}

TEST(CircuitParserTest, CircuitFromProgramInvalidQubitId) {
  Program program_proto;
  ::cirq::api::google::v2::Circuit* circuit_proto =
      program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  ::cirq::api::google::v2::Moment* moments_proto = circuit_proto->add_moments();

  // Add CNOT gate with invalid qubit
  ::cirq::api::google::v2::Operation* operations_proto =
      moments_proto->add_operations();
  ::cirq::api::google::v2::Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("CNOT");
  ::cirq::api::google::v2::Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");
  qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0_0");

  Circuit test_circuit;
  ASSERT_EQ(CircuitFromProgram(program_proto, 2, &test_circuit),
            tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                               "Could not parse Qubit id: " +
                                   qubits_proto->ShortDebugString()));
}

TEST(CircuitParserTest, CircuitFromProgramEmpty) {
  Program program_proto;
  ::cirq::api::google::v2::Circuit* circuit_proto =
      program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);

  Circuit real_circuit, test_circuit;
  // TODO(zaqqwerty): num_qubits <= 1 due to orphan gate collection method
  real_circuit.num_qubits = 0;
  CircuitFromProgram(program_proto, 0, &test_circuit);

  ASSERT_EQ(test_circuit.num_qubits, real_circuit.num_qubits);
  ASSERT_EQ(test_circuit.gates, real_circuit.gates);

  // Test application of orphan gate collection identity gates
  // TODO(zaqqwerty): remove when orphan gate collection is moved to fuser
  Program program_proto_ident_odd;
  ::cirq::api::google::v2::Circuit* circuit_proto_ident_odd =
      program_proto_ident_odd.mutable_circuit();
  circuit_proto_ident_odd->set_scheduling_strategy(
      circuit_proto_ident_odd->MOMENT_BY_MOMENT);
  Program program_proto_ident_even;
  ::cirq::api::google::v2::Circuit* circuit_proto_ident_even =
      program_proto_ident_even.mutable_circuit();
  circuit_proto_ident_even->set_scheduling_strategy(
      circuit_proto_ident_even->MOMENT_BY_MOMENT);

  I2GateBuilder ident_builder;
  std::vector<unsigned int> locations;
  absl::flat_hash_map<std::string, float> arg_map;
  Gate gate_01, gate_12, gate_23;
  locations.clear();
  locations.push_back(0);
  locations.push_back(1);
  ident_builder.Build(0, locations, arg_map, &gate_01);
  locations.clear();
  locations.push_back(1);
  locations.push_back(2);
  ident_builder.Build(1, locations, arg_map, &gate_12);
  locations.clear();
  locations.push_back(2);
  locations.push_back(3);
  ident_builder.Build(0, locations, arg_map, &gate_23);

  Circuit real_circuit_ident_odd, test_circuit_ident_odd;
  real_circuit_ident_odd.num_qubits = 3;
  real_circuit_ident_odd.gates.push_back(gate_01);
  real_circuit_ident_odd.gates.push_back(gate_12);
  CircuitFromProgram(program_proto_ident_odd, 3, &test_circuit_ident_odd);
  ASSERT_EQ(test_circuit_ident_odd, real_circuit_ident_odd);

  Circuit real_circuit_ident_even, test_circuit_ident_even;
  real_circuit_ident_even.num_qubits = 4;
  real_circuit_ident_even.gates.push_back(gate_01);
  real_circuit_ident_even.gates.push_back(gate_23);
  CircuitFromProgram(program_proto_ident_even, 4, &test_circuit_ident_even);
  ASSERT_EQ(test_circuit_ident_even, real_circuit_ident_even);
}

TEST(CircuitParserTest, CircuitFromProgramPaulis) {
  Program program_proto;
  ::cirq::api::google::v2::Circuit* circuit_proto =
      program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  ::cirq::api::google::v2::Moment* moments_proto = circuit_proto->add_moments();

  // Add X gate
  ::cirq::api::google::v2::Operation* operations_proto =
      moments_proto->add_operations();
  ::cirq::api::google::v2::Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("XP");
  ::cirq::api::google::v2::Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");

  // Create the required Arg protos
  ::cirq::api::google::v2::Arg global_shift_arg;
  ::cirq::api::google::v2::ArgValue* global_shift_arg_value =
      global_shift_arg.mutable_arg_value();
  global_shift_arg_value->set_float_value(0.0);
  ::cirq::api::google::v2::Arg exponent_arg;
  ::cirq::api::google::v2::ArgValue* exponent_arg_value =
      exponent_arg.mutable_arg_value();
  exponent_arg_value->set_float_value(1.0);
  ::cirq::api::google::v2::Arg exponent_scalar_arg;
  ::cirq::api::google::v2::ArgValue* exponent_scalar_arg_value =
      exponent_scalar_arg.mutable_arg_value();
  exponent_scalar_arg_value->set_float_value(1.0);

  // Add Arg protos to the operation arg map
  google::protobuf::Map<std::string, ::cirq::api::google::v2::Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["global_shift"] = global_shift_arg;
  (*args_proto)["exponent"] = exponent_arg;
  (*args_proto)["exponent_scalar"] = exponent_scalar_arg;

  ASSERT_EQ(program_proto.circuit().moments()[0].operations_size(), 1);

  // Build the corresponding correct circuit
  Circuit real_circuit;
  std::vector<unsigned int> locations;
  XPowGateBuilder builder;
  Gate gate_x;
  ::tensorflow::Status status;

  real_circuit.num_qubits = 1;
  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = 0.0;
  arg_map["exponent"] = 1.0;
  arg_map["exponent_scalar"] = 1.0;
  locations.push_back(real_circuit.num_qubits - 0 - 1);
  status = builder.Build(0, locations, arg_map, &gate_x);
  ASSERT_EQ(status, tensorflow::Status::OK());
  real_circuit.gates.push_back(gate_x);
  locations.clear();

  // Check conversion
  Circuit test_circuit;
  status = CircuitFromProgram(program_proto, 1, &test_circuit);
  ASSERT_EQ(status, tensorflow::Status::OK());
  ASSERT_EQ(test_circuit, real_circuit);
}

TEST(CircuitParserTest, CircuitFromPauliTermEmpty) {
  tfq::proto::PauliTerm pauli_proto;
  tensorflow::Status status;
  Circuit test_circuit, real_circuit;
  real_circuit.num_qubits = 0;
  status = CircuitFromPauliTerm(pauli_proto, 0, &test_circuit);
  ASSERT_EQ(status, tensorflow::Status::OK());
  ASSERT_EQ(test_circuit, real_circuit);
}

TEST(CircuitParserTest, CircuitFromPauliTermPauli) {
  tfq::proto::PauliTerm pauli_proto;
  // The created circuit should not depend on the coefficient
  pauli_proto.set_coefficient_real(3.14);
  tfq::proto::PauliQubitPair* pair_proto = pauli_proto.add_paulis();
  pair_proto->set_qubit_id("0");
  pair_proto->set_pauli_type("X");

  // Build the corresponding correct circuit
  Circuit real_circuit;
  std::vector<unsigned int> locations;
  XPowGateBuilder builder;
  Gate gate_x;
  ::tensorflow::Status status;

  real_circuit.num_qubits = 1;
  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = 0.0;
  arg_map["exponent"] = 1.0;
  arg_map["exponent_scalar"] = 1.0;
  locations.push_back(real_circuit.num_qubits - 0 - 1);
  status = builder.Build(0, locations, arg_map, &gate_x);
  ASSERT_EQ(status, tensorflow::Status::OK());
  real_circuit.gates.push_back(gate_x);
  locations.clear();

  // Check conversion
  Circuit test_circuit;
  status = CircuitFromPauliTerm(pauli_proto, 1, &test_circuit);
  ASSERT_EQ(status, tensorflow::Status::OK());
  ASSERT_EQ(test_circuit, real_circuit);
}

}  // namespace
}  // namespace tfq
