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

#include <string>

#include "../qsim/lib/channel.h"
#include "../qsim/lib/channels_cirq.h"
#include "../qsim/lib/circuit.h"
#include "../qsim/lib/circuit_noisy.h"
#include "../qsim/lib/gates_cirq.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/proto/program.pb.h"

namespace tfq {
namespace {

typedef absl::flat_hash_map<std::string, std::pair<int, float>> SymbolMap;
typedef qsim::Cirq::Channel<float> QsimChannel;
typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;
typedef qsim::NoisyCircuit<QsimGate> NoisyQsimCircuit;

using ::tfq::proto::Arg;
using ::tfq::proto::Circuit;
using ::tfq::proto::Gate;
using ::tfq::proto::Moment;
using ::tfq::proto::Operation;
using ::tfq::proto::Program;
using ::tfq::proto::Qubit;

Arg MakeArg(float val) {
  Arg arg;
  arg.mutable_arg_value()->set_float_value(val);
  return arg;
}

Arg MakeArg(const std::string& val) {
  Arg arg;
  arg.set_symbol(val);
  return arg;
}

Arg MakeControlArg(const std::string& val) {
  Arg arg;
  arg.mutable_arg_value()->set_string_value(val);
  return arg;
}

inline void AssertControlEqual(const QsimGate& a, const QsimGate& b) {
  for (std::vector<unsigned int>::size_type i = 0; i < a.controlled_by.size();
       i++) {
    ASSERT_EQ(a.controlled_by[i], b.controlled_by[i]);
  }
  ASSERT_EQ(a.cmask, b.cmask);
}

inline void AssertTwoQubitEqual(const QsimGate& a, const QsimGate& b) {
  for (int i = 0; i < 32; i++) {
    ASSERT_NEAR(a.matrix[i], b.matrix[i], 1e-5);
  }
  ASSERT_EQ(a.qubits[0], b.qubits[0]);
  ASSERT_EQ(a.qubits[1], b.qubits[1]);
  AssertControlEqual(a, b);
}

inline void AssertOneQubitEqual(const QsimGate& a, const QsimGate& b) {
  for (int i = 0; i < 8; i++) {
    ASSERT_NEAR(a.matrix[i], b.matrix[i], 1e-5);
  }
  ASSERT_EQ(a.qubits[0], b.qubits[0]);
  AssertControlEqual(a, b);
}

inline void AssertChannelEqual(const QsimChannel& a, const QsimChannel& b) {
  ASSERT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); i++) {
    ASSERT_EQ(a[i].kind, b[i].kind);
    ASSERT_EQ(a[i].unitary, b[i].unitary);
    ASSERT_NEAR(a[i].prob, b[i].prob, 1e-5);
    auto a_k_ops = a[i].ops;
    auto b_k_ops = b[i].ops;
    EXPECT_EQ(a_k_ops.size(), b_k_ops.size());
    for (int j = 0; j < a_k_ops.size(); j++) {
      AssertOneQubitEqual(a_k_ops[j], b_k_ops[j]);
    }
  }
}

class TwoQubitEigenFixture
    : public ::testing::TestWithParam<std::tuple<
          std::string, std::function<QsimGate(unsigned int, unsigned int,
                                              unsigned int, float, float)>>> {};

TEST_P(TwoQubitEigenFixture, TwoEigenGate) {
  float exp = 1.1234;
  float gs = 2.2345;

  // Get gate name and reference qsim gate.
  std::string name = std::get<0>(GetParam());
  auto ref_gate = std::get<1>(GetParam())(0, 1, 0, exp, gs);
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add gate.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id(name);

  // Set args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["global_shift"] = MakeArg(gs);
  (*args_proto)["exponent"] = MakeArg("placeholder");
  (*args_proto)["exponent_scalar"] = MakeArg(0.5);

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("");
  (*args_proto)["control_values"] = MakeControlArg("");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");
  qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("1");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap symbol_map = {{"placeholder", std::pair<int, float>(1, 2 * exp)}};
  std::vector<GateMetaData> metadata;

  // Test case where we have a placeholder.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status::OK());
  AssertTwoQubitEqual(test_circuit.gates[0], ref_gate);
  EXPECT_EQ(metadata[0].index, 0);
  EXPECT_EQ(metadata[0].symbol_values[0], "placeholder");
  EXPECT_EQ(metadata[0].placeholder_names[0], GateParamNames::kExponent);
  EXPECT_NEAR(metadata[0].gate_params[0], 2 * exp, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[1], 0.5, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[2], gs, 1e-5);
  EXPECT_EQ(metadata.size(), 1);
  EXPECT_EQ(metadata[0].gate_params.size(), 3);
  EXPECT_EQ(metadata[0].symbol_values.size(), 1);
  EXPECT_EQ(metadata[0].placeholder_names.size(), 1);

  test_circuit.gates.clear();
  fused_circuit.clear();
  metadata.clear();
  (*args_proto)["exponent"] = MakeArg(exp);
  (*args_proto)["exponent_scalar"] = MakeArg(1.0);
  symbol_map.clear();

  // Test case where we have all float values.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status::OK());
  AssertTwoQubitEqual(test_circuit.gates[0], ref_gate);
  EXPECT_EQ(metadata[0].index, 0);
  EXPECT_NEAR(metadata[0].gate_params[0], exp, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[1], 1.0, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[2], gs, 1e-5);
  EXPECT_EQ(metadata.size(), 1);
  EXPECT_EQ(metadata[0].gate_params.size(), 3);
  EXPECT_EQ(metadata[0].symbol_values.size(), 0);
  EXPECT_EQ(metadata[0].placeholder_names.size(), 0);

  test_circuit.gates.clear();
  fused_circuit.clear();
  (*args_proto).erase(args_proto->find("exponent"));
  symbol_map.clear();

  // Test case where proto arg missing.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                               "Could not find arg: exponent in op."));

  test_circuit.gates.clear();
  fused_circuit.clear();
  (*args_proto)["exponent"] = MakeArg("alpha");
  symbol_map.clear();

  // Test case where symbol value not present in resolver.
  ASSERT_EQ(
      QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                             &fused_circuit),
      tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                         "Could not find symbol in parameter map: alpha"));
}

TEST_P(TwoQubitEigenFixture, TwoEigenGateControlled) {
  float exp = 1.1234;
  float gs = 2.2345;

  // Get gate name and reference qsim gate.
  std::string name = std::get<0>(GetParam());
  auto ref_gate =
      std::get<1>(GetParam())(0, 1, 0, exp, gs).ControlledBy({2, 3}, {0, 0});
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add gate.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id(name);

  // Set args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["global_shift"] = MakeArg(gs);
  (*args_proto)["exponent"] = MakeArg("placeholder");
  (*args_proto)["exponent_scalar"] = MakeArg(0.5);

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("1,0");
  (*args_proto)["control_values"] = MakeControlArg("0,0");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("2");
  qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("3");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap symbol_map = {{"placeholder", std::pair<int, float>(1, 2 * exp)}};
  std::vector<GateMetaData> metadata;

  // Test case where we have a placeholder.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 4, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status::OK());
  AssertTwoQubitEqual(test_circuit.gates[0], ref_gate);
  EXPECT_EQ(metadata[0].index, 0);
  EXPECT_EQ(metadata[0].symbol_values[0], "placeholder");
  EXPECT_EQ(metadata[0].placeholder_names[0], GateParamNames::kExponent);
  EXPECT_NEAR(metadata[0].gate_params[0], 2 * exp, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[1], 0.5, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[2], gs, 1e-5);
  EXPECT_EQ(metadata.size(), 1);
  EXPECT_EQ(metadata[0].gate_params.size(), 3);
  EXPECT_EQ(metadata[0].symbol_values.size(), 1);
  EXPECT_EQ(metadata[0].placeholder_names.size(), 1);
}

INSTANTIATE_TEST_CASE_P(
    TwoQubitEigenTests, TwoQubitEigenFixture,
    ::testing::Values(
        std::make_tuple("CNP", &qsim::Cirq::CXPowGate<float>::Create),
        std::make_tuple("CZP", &qsim::Cirq::CZPowGate<float>::Create),
        std::make_tuple("SP", &qsim::Cirq::SwapPowGate<float>::Create),
        std::make_tuple("ISP", &qsim::Cirq::ISwapPowGate<float>::Create),
        std::make_tuple("XXP", &qsim::Cirq::XXPowGate<float>::Create),
        std::make_tuple("YYP", &qsim::Cirq::YYPowGate<float>::Create),
        std::make_tuple("ZZP", &qsim::Cirq::ZZPowGate<float>::Create)));

class SingleQubitEigenFixture
    : public ::testing::TestWithParam<std::tuple<
          std::string, std::function<QsimGate(int, int, float, float)>>> {};

TEST_P(SingleQubitEigenFixture, SingleEigenGate) {
  float exp = 1.1234;
  float gs = 2.2345;

  // Get gate name and reference qsim gate.
  std::string name = std::get<0>(GetParam());
  auto ref_gate = std::get<1>(GetParam())(0, 0, exp, gs);
  // Try symbol resolution.
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add gate.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id(name);

  // Set args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["global_shift"] = MakeArg(gs);
  (*args_proto)["exponent"] = MakeArg("placeholder");
  (*args_proto)["exponent_scalar"] = MakeArg(0.5);

  // Set the control args to empty.
  (*args_proto)["control_qubits"] = MakeControlArg("");
  (*args_proto)["control_values"] = MakeControlArg("");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap symbol_map = {{"placeholder", std::pair<int, float>(1, 2 * exp)}};
  std::vector<GateMetaData> metadata;

  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 1, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status::OK());
  AssertOneQubitEqual(test_circuit.gates[0], ref_gate);
  EXPECT_EQ(metadata[0].index, 0);
  EXPECT_EQ(metadata[0].symbol_values[0], "placeholder");
  EXPECT_EQ(metadata[0].placeholder_names[0], GateParamNames::kExponent);
  EXPECT_NEAR(metadata[0].gate_params[0], 2 * exp, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[1], 0.5, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[2], gs, 1e-5);
  EXPECT_EQ(metadata.size(), 1);
  EXPECT_EQ(metadata[0].gate_params.size(), 3);
  EXPECT_EQ(metadata[0].symbol_values.size(), 1);
  EXPECT_EQ(metadata[0].placeholder_names.size(), 1);

  test_circuit.gates.clear();
  fused_circuit.clear();
  (*args_proto)["exponent"] = MakeArg(exp);
  (*args_proto)["exponent_scalar"] = MakeArg(1.0);
  symbol_map.clear();
  metadata.clear();

  // Test case where we have all float values.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 1, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status::OK());
  AssertOneQubitEqual(test_circuit.gates[0], ref_gate);
  EXPECT_EQ(metadata[0].index, 0);
  EXPECT_NEAR(metadata[0].gate_params[0], exp, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[1], 1.0, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[2], gs, 1e-5);
  EXPECT_EQ(metadata.size(), 1);
  EXPECT_EQ(metadata[0].gate_params.size(), 3);
  EXPECT_EQ(metadata[0].symbol_values.size(), 0);
  EXPECT_EQ(metadata[0].placeholder_names.size(), 0);

  test_circuit.gates.clear();
  fused_circuit.clear();
  (*args_proto).erase(args_proto->find("exponent"));
  symbol_map.clear();

  // Test case where proto arg missing.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 1, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                               "Could not find arg: exponent in op."));

  test_circuit.gates.clear();
  fused_circuit.clear();
  (*args_proto)["exponent"] = MakeArg("alpha");
  symbol_map.clear();

  // Test case where symbol value not present in resolver.
  ASSERT_EQ(
      QsimCircuitFromProgram(program_proto, symbol_map, 1, &test_circuit,
                             &fused_circuit),
      tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                         "Could not find symbol in parameter map: alpha"));
}

TEST_P(SingleQubitEigenFixture, SingleEigenGateControlled) {
  float exp = 1.1234;
  float gs = 2.2345;

  // Get gate name and reference qsim gate.
  std::string name = std::get<0>(GetParam());
  auto ref_gate =
      std::get<1>(GetParam())(0, 0, exp, gs).ControlledBy({1, 2}, {0, 0});
  // Try symbol resolution.
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add gate.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id(name);

  // Set args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["global_shift"] = MakeArg(gs);
  (*args_proto)["exponent"] = MakeArg("placeholder");
  (*args_proto)["exponent_scalar"] = MakeArg(0.5);

  // Set the control args to empty.
  (*args_proto)["control_qubits"] = MakeControlArg("1,0");
  (*args_proto)["control_values"] = MakeControlArg("0,0");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("2");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap symbol_map = {{"placeholder", std::pair<int, float>(1, 2 * exp)}};
  std::vector<GateMetaData> metadata;

  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 3, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status::OK());
  AssertOneQubitEqual(test_circuit.gates[0], ref_gate);
  EXPECT_EQ(metadata[0].index, 0);
  EXPECT_EQ(metadata[0].symbol_values[0], "placeholder");
  EXPECT_EQ(metadata[0].placeholder_names[0], GateParamNames::kExponent);
  EXPECT_NEAR(metadata[0].gate_params[0], 2 * exp, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[1], 0.5, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[2], gs, 1e-5);
  EXPECT_EQ(metadata.size(), 1);
  EXPECT_EQ(metadata[0].gate_params.size(), 3);
  EXPECT_EQ(metadata[0].symbol_values.size(), 1);
  EXPECT_EQ(metadata[0].placeholder_names.size(), 1);
}

INSTANTIATE_TEST_CASE_P(
    SingleQubitEigenTests, SingleQubitEigenFixture,
    ::testing::Values(
        std::make_tuple("HP", &qsim::Cirq::HPowGate<float>::Create),
        std::make_tuple("XP", &qsim::Cirq::XPowGate<float>::Create),
        std::make_tuple("YP", &qsim::Cirq::YPowGate<float>::Create),
        std::make_tuple("ZP", &qsim::Cirq::ZPowGate<float>::Create)));

TEST(QsimCircuitParserTest, SingleConstantGate) {
  absl::flat_hash_map<std::string, QsimGate> reference = {
      {"I", qsim::Cirq::I1<float>::Create(0, 0)}};
  for (auto kv : reference) {
    Program program_proto;
    Circuit* circuit_proto = program_proto.mutable_circuit();
    circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
    Moment* moments_proto = circuit_proto->add_moments();

    // Add gate.
    Operation* operations_proto = moments_proto->add_operations();
    Gate* gate_proto = operations_proto->mutable_gate();
    gate_proto->set_id(kv.first);

    // Set the control args to empty.
    google::protobuf::Map<std::string, Arg>* args_proto =
        operations_proto->mutable_args();
    (*args_proto)["control_qubits"] = MakeControlArg("");
    (*args_proto)["control_values"] = MakeControlArg("");

    // Set the qubits.
    Qubit* qubits_proto = operations_proto->add_qubits();
    qubits_proto->set_id("0");

    QsimCircuit test_circuit;
    std::vector<qsim::GateFused<QsimGate>> fused_circuit;
    SymbolMap empty_map;
    std::vector<GateMetaData> metadata;

    ASSERT_EQ(QsimCircuitFromProgram(program_proto, empty_map, 1, &test_circuit,
                                     &fused_circuit, &metadata),
              tensorflow::Status::OK());
    AssertOneQubitEqual(test_circuit.gates[0], kv.second);
    EXPECT_EQ(metadata.size(), 1);
    EXPECT_EQ(metadata[0].placeholder_names.size(), 0);
    EXPECT_EQ(metadata[0].symbol_values.size(), 0);
    EXPECT_EQ(metadata[0].gate_params.size(), 0);
  }
}

TEST(QsimCircuitParserTest, SingleConstantGateControlled) {
  absl::flat_hash_map<std::string, QsimGate> reference = {
      {"I", qsim::Cirq::I1<float>::Create(0, 0).ControlledBy({1, 2}, {0, 0})}};
  for (auto kv : reference) {
    Program program_proto;
    Circuit* circuit_proto = program_proto.mutable_circuit();
    circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
    Moment* moments_proto = circuit_proto->add_moments();

    // Add gate.
    Operation* operations_proto = moments_proto->add_operations();
    Gate* gate_proto = operations_proto->mutable_gate();
    gate_proto->set_id(kv.first);

    // Set the control args to empty.
    google::protobuf::Map<std::string, Arg>* args_proto =
        operations_proto->mutable_args();
    (*args_proto)["control_qubits"] = MakeControlArg("1,0");
    (*args_proto)["control_values"] = MakeControlArg("0,0");

    // Set the qubits.
    Qubit* qubits_proto = operations_proto->add_qubits();
    qubits_proto->set_id("2");

    QsimCircuit test_circuit;
    std::vector<qsim::GateFused<QsimGate>> fused_circuit;
    SymbolMap empty_map;
    std::vector<GateMetaData> metadata;

    ASSERT_EQ(QsimCircuitFromProgram(program_proto, empty_map, 3, &test_circuit,
                                     &fused_circuit, &metadata),
              tensorflow::Status::OK());
    AssertOneQubitEqual(test_circuit.gates[0], kv.second);
    EXPECT_EQ(metadata.size(), 1);
    EXPECT_EQ(metadata[0].placeholder_names.size(), 0);
    EXPECT_EQ(metadata[0].symbol_values.size(), 0);
    EXPECT_EQ(metadata[0].gate_params.size(), 0);
  }
}

TEST(QsimCircuitParserTest, TwoConstantGate) {
  absl::flat_hash_map<std::string, QsimGate> reference = {
      {"I2", qsim::Cirq::I2<float>::Create(0, 1, 0)}};
  for (auto kv : reference) {
    Program program_proto;
    Circuit* circuit_proto = program_proto.mutable_circuit();
    circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
    Moment* moments_proto = circuit_proto->add_moments();

    // Add gate.
    Operation* operations_proto = moments_proto->add_operations();
    Gate* gate_proto = operations_proto->mutable_gate();
    gate_proto->set_id(kv.first);

    // Set the control args to empty.
    google::protobuf::Map<std::string, Arg>* args_proto =
        operations_proto->mutable_args();
    (*args_proto)["control_qubits"] = MakeControlArg("");
    (*args_proto)["control_values"] = MakeControlArg("");

    // Set the qubits.
    Qubit* qubits_proto = operations_proto->add_qubits();
    qubits_proto->set_id("0");
    qubits_proto = operations_proto->add_qubits();
    qubits_proto->set_id("1");

    QsimCircuit test_circuit;
    std::vector<qsim::GateFused<QsimGate>> fused_circuit;
    SymbolMap empty_map;
    std::vector<GateMetaData> metadata;

    ASSERT_EQ(QsimCircuitFromProgram(program_proto, empty_map, 2, &test_circuit,
                                     &fused_circuit, &metadata),
              tensorflow::Status::OK());
    AssertTwoQubitEqual(test_circuit.gates[0], kv.second);
    EXPECT_EQ(metadata.size(), 1);
    EXPECT_EQ(metadata[0].placeholder_names.size(), 0);
    EXPECT_EQ(metadata[0].symbol_values.size(), 0);
    EXPECT_EQ(metadata[0].gate_params.size(), 0);
  }
}

TEST(QsimCircuitParserTest, TwoConstantGateControlled) {
  absl::flat_hash_map<std::string, QsimGate> reference = {
      {"I2",
       qsim::Cirq::I2<float>::Create(0, 1, 0).ControlledBy({2, 3}, {0, 0})}};
  for (auto kv : reference) {
    Program program_proto;
    Circuit* circuit_proto = program_proto.mutable_circuit();
    circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
    Moment* moments_proto = circuit_proto->add_moments();

    // Add gate.
    Operation* operations_proto = moments_proto->add_operations();
    Gate* gate_proto = operations_proto->mutable_gate();
    gate_proto->set_id(kv.first);

    // Set the control args to empty.
    google::protobuf::Map<std::string, Arg>* args_proto =
        operations_proto->mutable_args();
    (*args_proto)["control_qubits"] = MakeControlArg("1,0");
    (*args_proto)["control_values"] = MakeControlArg("0,0");

    // Set the qubits.
    Qubit* qubits_proto = operations_proto->add_qubits();
    qubits_proto->set_id("2");
    qubits_proto = operations_proto->add_qubits();
    qubits_proto->set_id("3");

    QsimCircuit test_circuit;
    std::vector<qsim::GateFused<QsimGate>> fused_circuit;
    SymbolMap empty_map;
    std::vector<GateMetaData> metadata;

    ASSERT_EQ(QsimCircuitFromProgram(program_proto, empty_map, 4, &test_circuit,
                                     &fused_circuit, &metadata),
              tensorflow::Status::OK());
    AssertTwoQubitEqual(test_circuit.gates[0], kv.second);
    EXPECT_EQ(metadata.size(), 1);
    EXPECT_EQ(metadata[0].placeholder_names.size(), 0);
    EXPECT_EQ(metadata[0].symbol_values.size(), 0);
    EXPECT_EQ(metadata[0].gate_params.size(), 0);
  }
}

TEST(QsimCircuitParserTest, FsimGate) {
  float theta = 0.1234;
  float phi = 0.4567;
  auto reference = qsim::Cirq::FSimGate<float>::Create(0, 0, 1, theta, phi);
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add gate.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("FSIM");

  // Set the args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["theta"] = MakeArg("alpha");
  (*args_proto)["theta_scalar"] = MakeArg(0.5);
  (*args_proto)["phi"] = MakeArg("beta");
  (*args_proto)["phi_scalar"] = MakeArg(0.2);

  // Set the control args to empty.
  (*args_proto)["control_qubits"] = MakeControlArg("");
  (*args_proto)["control_values"] = MakeControlArg("");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");
  qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("1");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap symbol_map = {{"alpha", std::pair<int, float>(0, 2 * theta)},
                          {"beta", std::pair<int, float>(1, 5 * phi)}};
  std::vector<GateMetaData> metadata;

  // Test symbol resolution.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status::OK());
  AssertTwoQubitEqual(test_circuit.gates[0], reference);
  EXPECT_EQ(metadata.size(), 1);
  EXPECT_EQ(metadata[0].placeholder_names.size(), 2);
  EXPECT_EQ(metadata[0].symbol_values.size(), 2);
  EXPECT_EQ(metadata[0].gate_params.size(), 4);
  EXPECT_NEAR(metadata[0].gate_params[0], 2 * theta, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[1], 0.5, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[2], 5 * phi, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[3], 0.2, 1e-5);
  EXPECT_EQ(metadata[0].symbol_values[0], "alpha");
  EXPECT_EQ(metadata[0].symbol_values[1], "beta");
  EXPECT_EQ(metadata[0].placeholder_names[0], GateParamNames::kTheta);
  EXPECT_EQ(metadata[0].placeholder_names[1], GateParamNames::kPhi);

  symbol_map.clear();
  test_circuit.gates.clear();
  fused_circuit.clear();
  metadata.clear();
  (*args_proto)["theta"] = MakeArg(theta);
  (*args_proto)["theta_scalar"] = MakeArg(1.0);
  (*args_proto)["phi"] = MakeArg(phi);
  (*args_proto)["phi_scalar"] = MakeArg(1.0);

  // Test float values only.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status::OK());
  AssertTwoQubitEqual(test_circuit.gates[0], reference);
  EXPECT_EQ(metadata.size(), 1);
  EXPECT_EQ(metadata[0].placeholder_names.size(), 0);
  EXPECT_EQ(metadata[0].symbol_values.size(), 0);
  EXPECT_EQ(metadata[0].gate_params.size(), 4);
  EXPECT_NEAR(metadata[0].gate_params[0], theta, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[1], 1.0, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[2], phi, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[3], 1.0, 1e-5);

  test_circuit.gates.clear();
  fused_circuit.clear();
  args_proto->erase(args_proto->find("theta"));

  // Test case where proto arg missing.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                               "Could not find arg: theta in op."));

  test_circuit.gates.clear();
  fused_circuit.clear();
  (*args_proto)["theta"] = MakeArg("alpha");
  symbol_map.clear();

  // Test case where symbol value not present in resolver.
  ASSERT_EQ(
      QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                             &fused_circuit),
      tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                         "Could not find symbol in parameter map: alpha"));
}

TEST(QsimCircuitParserTest, FsimGateControlled) {
  float theta = 0.1234;
  float phi = 0.4567;
  auto reference = qsim::Cirq::FSimGate<float>::Create(0, 0, 1, theta, phi)
                       .ControlledBy({2, 3}, {0, 0});
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add gate.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("FSIM");

  // Set the args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["theta"] = MakeArg("alpha");
  (*args_proto)["theta_scalar"] = MakeArg(0.5);
  (*args_proto)["phi"] = MakeArg("beta");
  (*args_proto)["phi_scalar"] = MakeArg(0.2);

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("1,0");
  (*args_proto)["control_values"] = MakeControlArg("0,0");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("2");
  qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("3");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap symbol_map = {{"alpha", std::pair<int, float>(0, 2 * theta)},
                          {"beta", std::pair<int, float>(1, 5 * phi)}};
  std::vector<GateMetaData> metadata;

  // Test symbol resolution.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 4, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status::OK());
  AssertTwoQubitEqual(test_circuit.gates[0], reference);
  EXPECT_EQ(metadata.size(), 1);
  EXPECT_EQ(metadata[0].placeholder_names.size(), 2);
  EXPECT_EQ(metadata[0].symbol_values.size(), 2);
  EXPECT_EQ(metadata[0].gate_params.size(), 4);
  EXPECT_NEAR(metadata[0].gate_params[0], 2 * theta, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[1], 0.5, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[2], 5 * phi, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[3], 0.2, 1e-5);
  EXPECT_EQ(metadata[0].symbol_values[0], "alpha");
  EXPECT_EQ(metadata[0].symbol_values[1], "beta");
  EXPECT_EQ(metadata[0].placeholder_names[0], GateParamNames::kTheta);
  EXPECT_EQ(metadata[0].placeholder_names[1], GateParamNames::kPhi);
}

TEST(QsimCircuitParserTest, PhasedISwap) {
  float exponent = 0.1234;
  float phase_exponent = 0.4567;
  auto reference = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, 1, 0, phase_exponent, exponent);
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add gate.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("PISP");

  // Set the args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["phase_exponent"] = MakeArg("alpha");
  (*args_proto)["phase_exponent_scalar"] = MakeArg(0.5);
  (*args_proto)["exponent"] = MakeArg("beta");
  (*args_proto)["exponent_scalar"] = MakeArg(0.2);

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("");
  (*args_proto)["control_values"] = MakeControlArg("");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");
  qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("1");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap symbol_map = {
      {"alpha", std::pair<int, float>(0, 2 * phase_exponent)},
      {"beta", std::pair<int, float>(1, 5 * exponent)}};
  std::vector<GateMetaData> metadata;

  // Test symbol resolution.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status::OK());
  AssertTwoQubitEqual(test_circuit.gates[0], reference);
  EXPECT_EQ(metadata.size(), 1);
  EXPECT_EQ(metadata[0].placeholder_names.size(), 2);
  EXPECT_EQ(metadata[0].symbol_values.size(), 2);
  EXPECT_EQ(metadata[0].gate_params.size(), 4);
  EXPECT_NEAR(metadata[0].gate_params[0], 2 * phase_exponent, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[1], 0.5, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[2], 5 * exponent, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[3], 0.2, 1e-5);
  EXPECT_EQ(metadata[0].symbol_values[0], "alpha");
  EXPECT_EQ(metadata[0].symbol_values[1], "beta");
  EXPECT_EQ(metadata[0].placeholder_names[0], GateParamNames::kPhaseExponent);
  EXPECT_EQ(metadata[0].placeholder_names[1], GateParamNames::kExponent);

  symbol_map.clear();
  test_circuit.gates.clear();
  fused_circuit.clear();
  metadata.clear();
  (*args_proto)["phase_exponent"] = MakeArg(phase_exponent);
  (*args_proto)["phase_exponent_scalar"] = MakeArg(1.0);
  (*args_proto)["exponent"] = MakeArg(exponent);
  (*args_proto)["exponent_scalar"] = MakeArg(1.0);

  // Test float values only.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status::OK());
  AssertTwoQubitEqual(test_circuit.gates[0], reference);
  EXPECT_EQ(metadata.size(), 1);
  EXPECT_EQ(metadata[0].placeholder_names.size(), 0);
  EXPECT_EQ(metadata[0].symbol_values.size(), 0);
  EXPECT_EQ(metadata[0].gate_params.size(), 4);
  EXPECT_NEAR(metadata[0].gate_params[0], phase_exponent, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[1], 1.0, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[2], exponent, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[3], 1.0, 1e-5);

  test_circuit.gates.clear();
  fused_circuit.clear();
  args_proto->erase(args_proto->find("phase_exponent"));

  // Test case where proto arg missing.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                               "Could not find arg: phase_exponent in op."));

  test_circuit.gates.clear();
  fused_circuit.clear();
  (*args_proto)["phase_exponent"] = MakeArg("alpha");
  symbol_map.clear();

  // Test case where symbol value not present in resolver.
  ASSERT_EQ(
      QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                             &fused_circuit),
      tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                         "Could not find symbol in parameter map: alpha"));
}

TEST(QsimCircuitParserTest, PhasedISwapControlled) {
  float exponent = 0.1234;
  float phase_exponent = 0.4567;
  auto reference = qsim::Cirq::PhasedISwapPowGate<float>::Create(
                       0, 1, 0, phase_exponent, exponent)
                       .ControlledBy({2, 3}, {0, 0});
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add gate.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("PISP");

  // Set the args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["phase_exponent"] = MakeArg("alpha");
  (*args_proto)["phase_exponent_scalar"] = MakeArg(0.5);
  (*args_proto)["exponent"] = MakeArg("beta");
  (*args_proto)["exponent_scalar"] = MakeArg(0.2);

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("1,0");
  (*args_proto)["control_values"] = MakeControlArg("0,0");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("2");
  qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("3");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap symbol_map = {
      {"alpha", std::pair<int, float>(0, 2 * phase_exponent)},
      {"beta", std::pair<int, float>(1, 5 * exponent)}};
  std::vector<GateMetaData> metadata;

  // Test symbol resolution.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 4, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status::OK());
  AssertTwoQubitEqual(test_circuit.gates[0], reference);
  EXPECT_EQ(metadata.size(), 1);
  EXPECT_EQ(metadata[0].placeholder_names.size(), 2);
  EXPECT_EQ(metadata[0].symbol_values.size(), 2);
  EXPECT_EQ(metadata[0].gate_params.size(), 4);
  EXPECT_NEAR(metadata[0].gate_params[0], 2 * phase_exponent, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[1], 0.5, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[2], 5 * exponent, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[3], 0.2, 1e-5);
  EXPECT_EQ(metadata[0].symbol_values[0], "alpha");
  EXPECT_EQ(metadata[0].symbol_values[1], "beta");
  EXPECT_EQ(metadata[0].placeholder_names[0], GateParamNames::kPhaseExponent);
  EXPECT_EQ(metadata[0].placeholder_names[1], GateParamNames::kExponent);
}

TEST(QsimCircuitParserTest, PhasedXPow) {
  float exponent = 0.1234;
  float phase_exponent = 0.4567;
  float gs = 0.8910;
  auto reference = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, 0, phase_exponent, exponent, gs);
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add gate.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("PXP");

  // Set the args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["phase_exponent"] = MakeArg("alpha");
  (*args_proto)["phase_exponent_scalar"] = MakeArg(0.5);
  (*args_proto)["exponent"] = MakeArg("beta");
  (*args_proto)["exponent_scalar"] = MakeArg(0.2);
  (*args_proto)["global_shift"] = MakeArg(gs);

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("");
  (*args_proto)["control_values"] = MakeControlArg("");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap symbol_map = {
      {"alpha", std::pair<int, float>(0, 2 * phase_exponent)},
      {"beta", std::pair<int, float>(1, 5 * exponent)}};
  std::vector<GateMetaData> metadata;

  // Test symbol resolution.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 1, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status::OK());
  AssertOneQubitEqual(test_circuit.gates[0], reference);
  EXPECT_EQ(metadata.size(), 1);
  EXPECT_EQ(metadata[0].placeholder_names.size(), 2);
  EXPECT_EQ(metadata[0].symbol_values.size(), 2);
  EXPECT_EQ(metadata[0].gate_params.size(), 5);
  EXPECT_NEAR(metadata[0].gate_params[0], 2 * phase_exponent, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[1], 0.5, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[2], 5 * exponent, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[3], 0.2, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[4], gs, 1e-5);
  EXPECT_EQ(metadata[0].symbol_values[0], "alpha");
  EXPECT_EQ(metadata[0].symbol_values[1], "beta");
  EXPECT_EQ(metadata[0].placeholder_names[0], GateParamNames::kPhaseExponent);
  EXPECT_EQ(metadata[0].placeholder_names[1], GateParamNames::kExponent);

  symbol_map.clear();
  test_circuit.gates.clear();
  fused_circuit.clear();
  metadata.clear();
  (*args_proto)["phase_exponent"] = MakeArg(phase_exponent);
  (*args_proto)["phase_exponent_scalar"] = MakeArg(1.0);
  (*args_proto)["exponent"] = MakeArg(exponent);
  (*args_proto)["exponent_scalar"] = MakeArg(1.0);
  (*args_proto)["global_shift"] = MakeArg(gs);

  // Test float values only.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 1, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status::OK());
  AssertOneQubitEqual(test_circuit.gates[0], reference);
  EXPECT_EQ(metadata.size(), 1);
  EXPECT_EQ(metadata[0].placeholder_names.size(), 0);
  EXPECT_EQ(metadata[0].symbol_values.size(), 0);
  EXPECT_EQ(metadata[0].gate_params.size(), 5);
  EXPECT_NEAR(metadata[0].gate_params[0], phase_exponent, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[1], 1.0, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[2], exponent, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[3], 1.0, 1e-5);

  test_circuit.gates.clear();
  fused_circuit.clear();
  args_proto->erase(args_proto->find("phase_exponent"));

  // Test case where proto arg missing.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 1, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                               "Could not find arg: phase_exponent in op."));

  test_circuit.gates.clear();
  fused_circuit.clear();
  (*args_proto)["phase_exponent"] = MakeArg("alpha");
  symbol_map.clear();

  // Test case where symbol value not present in resolver.
  ASSERT_EQ(
      QsimCircuitFromProgram(program_proto, symbol_map, 1, &test_circuit,
                             &fused_circuit),
      tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                         "Could not find symbol in parameter map: alpha"));
}

TEST(QsimCircuitParserTest, PhasedXPowControlled) {
  float exponent = 0.1234;
  float phase_exponent = 0.4567;
  float gs = 0.8910;
  auto reference = qsim::Cirq::PhasedXPowGate<float>::Create(
                       0, 0, phase_exponent, exponent, gs)
                       .ControlledBy({1, 2}, {0, 0});
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add gate.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("PXP");

  // Set the args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["phase_exponent"] = MakeArg("alpha");
  (*args_proto)["phase_exponent_scalar"] = MakeArg(0.5);
  (*args_proto)["exponent"] = MakeArg("beta");
  (*args_proto)["exponent_scalar"] = MakeArg(0.2);
  (*args_proto)["global_shift"] = MakeArg(gs);

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("1,0");
  (*args_proto)["control_values"] = MakeControlArg("0,0");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("2");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap symbol_map = {
      {"alpha", std::pair<int, float>(0, 2 * phase_exponent)},
      {"beta", std::pair<int, float>(1, 5 * exponent)}};
  std::vector<GateMetaData> metadata;

  // Test symbol resolution.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 3, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status::OK());
  AssertOneQubitEqual(test_circuit.gates[0], reference);
  EXPECT_EQ(metadata.size(), 1);
  EXPECT_EQ(metadata[0].placeholder_names.size(), 2);
  EXPECT_EQ(metadata[0].symbol_values.size(), 2);
  EXPECT_EQ(metadata[0].gate_params.size(), 5);
  EXPECT_NEAR(metadata[0].gate_params[0], 2 * phase_exponent, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[1], 0.5, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[2], 5 * exponent, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[3], 0.2, 1e-5);
  EXPECT_NEAR(metadata[0].gate_params[4], gs, 1e-5);
  EXPECT_EQ(metadata[0].symbol_values[0], "alpha");
  EXPECT_EQ(metadata[0].symbol_values[1], "beta");
  EXPECT_EQ(metadata[0].placeholder_names[0], GateParamNames::kPhaseExponent);
  EXPECT_EQ(metadata[0].placeholder_names[1], GateParamNames::kExponent);
}

TEST(QsimCircuitParserTest, InvalidControlValues) {
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add gate.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("I");

  // Set the control args to empty.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["control_qubits"] = MakeControlArg("1,0");
  (*args_proto)["control_values"] = MakeControlArg("0,junk");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("2");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap empty_map;
  std::vector<GateMetaData> metadata;

  ASSERT_EQ(QsimCircuitFromProgram(program_proto, empty_map, 3, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                               "Unparseable control value: junk"));
}

TEST(QsimCircuitParserTest, MismatchControlNum) {
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add gate.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("I");

  // Set the control args to empty.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["control_qubits"] = MakeControlArg("1,0");
  (*args_proto)["control_values"] = MakeControlArg("0");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("2");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap empty_map;
  std::vector<GateMetaData> metadata;

  ASSERT_EQ(QsimCircuitFromProgram(program_proto, empty_map, 3, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status(
                tensorflow::error::INVALID_ARGUMENT,
                "Mistmatched number of control qubits and control values."));
}

TEST(QsimCircuitParserTest, EmptyTest) {
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap empty_map;
  std::vector<GateMetaData> metadata;

  // Ensure that nothing bad happens with an empty circuit.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, empty_map, 2, &test_circuit,
                                   &fused_circuit, &metadata),
            tensorflow::Status::OK());
  ASSERT_EQ(test_circuit.gates.size(), 0);
  ASSERT_EQ(fused_circuit.size(), 0);
  ASSERT_EQ(metadata.size(), 0);
}

TEST(QsimCircuitParserTest, CompoundCircuit) {
  float p = 0.1234;
  auto ref_chan = qsim::Cirq::DepolarizingChannel<float>::Create(0, 0, p);
  auto ref_gate = qsim::Cirq::I1<float>::Create(0, 1);
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add channel.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("DP");

  // Set the args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["p"] = MakeArg(p);

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("");
  (*args_proto)["control_values"] = MakeControlArg("");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("1");

  // Add gate.
  operations_proto = moments_proto->add_operations();
  gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("I");

  // Set the args.
  args_proto = operations_proto->mutable_args();

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("");
  (*args_proto)["control_values"] = MakeControlArg("");

  // Set the qubits.
  qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");

  NoisyQsimCircuit test_circuit;

  ASSERT_EQ(
      NoisyQsimCircuitFromProgram(program_proto, {}, 2, true, &test_circuit),
      tensorflow::Status::OK());
  AssertChannelEqual(test_circuit.channels[0], ref_chan);
  AssertOneQubitEqual(test_circuit.channels[1][0].ops[0], ref_gate);
  ASSERT_EQ(test_circuit.channels.size(),
            3);  // 2 gates + 1 layer of measurement.
  ASSERT_EQ(test_circuit.num_qubits, 2);
}

TEST(QsimCircuitParserTest, AsymmetricDepolarizing) {
  float p_x = 0.123;
  float p_y = 0.456;
  float p_z = 0.789;
  auto reference = qsim::Cirq::AsymmetricDepolarizingChannel<float>::Create(
      0, 0, p_x, p_y, p_z);
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add channel.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("ADP");

  // Set the args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["p_x"] = MakeArg(p_x);
  (*args_proto)["p_y"] = MakeArg(p_y);
  (*args_proto)["p_z"] = MakeArg(p_z);

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("");
  (*args_proto)["control_values"] = MakeControlArg("");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");

  NoisyQsimCircuit test_circuit;

  ASSERT_EQ(
      NoisyQsimCircuitFromProgram(program_proto, {}, 1, false, &test_circuit),
      tensorflow::Status::OK());
  AssertChannelEqual(test_circuit.channels[0], reference);
  ASSERT_EQ(test_circuit.channels.size(), 1);
  ASSERT_EQ(test_circuit.num_qubits, 1);
}

TEST(QsimCircuitParserTest, AmplitudeDamping) {
  float gamma = 0.1234;
  auto reference =
      qsim::Cirq::AmplitudeDampingChannel<float>::Create(0, 0, gamma);
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add channel.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("AD");

  // Set the args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["gamma"] = MakeArg(gamma);

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("");
  (*args_proto)["control_values"] = MakeControlArg("");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");

  NoisyQsimCircuit test_circuit;

  ASSERT_EQ(
      NoisyQsimCircuitFromProgram(program_proto, {}, 1, false, &test_circuit),
      tensorflow::Status::OK());
  AssertChannelEqual(test_circuit.channels[0], reference);
  ASSERT_EQ(test_circuit.channels.size(), 1);
  ASSERT_EQ(test_circuit.num_qubits, 1);
}

TEST(QsimCircuitParserTest, Depolarizing) {
  float p = 0.1234;
  auto reference = qsim::Cirq::DepolarizingChannel<float>::Create(0, 0, p);
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add channel.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("DP");

  // Set the args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["p"] = MakeArg(p);

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("");
  (*args_proto)["control_values"] = MakeControlArg("");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");

  NoisyQsimCircuit test_circuit;

  ASSERT_EQ(
      NoisyQsimCircuitFromProgram(program_proto, {}, 1, false, &test_circuit),
      tensorflow::Status::OK());
  AssertChannelEqual(test_circuit.channels[0], reference);
  ASSERT_EQ(test_circuit.channels.size(), 1);
  ASSERT_EQ(test_circuit.num_qubits, 1);
}

TEST(QsimCircuitParserTest, GeneralizedAmplitudeDamping) {
  float p = 0.123;
  float gamma = 0.456;
  auto reference =
      qsim::Cirq::GeneralizedAmplitudeDampingChannel<float>::Create(0, 0, p,
                                                                    gamma);
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add channel.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("GAD");

  // Set the args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["p"] = MakeArg(p);
  (*args_proto)["gamma"] = MakeArg(gamma);

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("");
  (*args_proto)["control_values"] = MakeControlArg("");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");

  NoisyQsimCircuit test_circuit;

  ASSERT_EQ(
      NoisyQsimCircuitFromProgram(program_proto, {}, 1, false, &test_circuit),
      tensorflow::Status::OK());
  AssertChannelEqual(test_circuit.channels[0], reference);
  ASSERT_EQ(test_circuit.channels.size(), 1);
  ASSERT_EQ(test_circuit.num_qubits, 1);
}

TEST(QsimCircuitParserTest, Reset) {
  auto reference = qsim::Cirq::ResetChannel<float>::Create(0, 0);
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add channel.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("RST");

  // Set the args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("");
  (*args_proto)["control_values"] = MakeControlArg("");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");

  NoisyQsimCircuit test_circuit;

  ASSERT_EQ(
      NoisyQsimCircuitFromProgram(program_proto, {}, 1, false, &test_circuit),
      tensorflow::Status::OK());
  AssertChannelEqual(test_circuit.channels[0], reference);
  ASSERT_EQ(test_circuit.channels.size(), 1);
  ASSERT_EQ(test_circuit.num_qubits, 1);
}

TEST(QsimCircuitParserTest, PhaseDamping) {
  float gamma = 0.1234;
  auto reference = qsim::Cirq::PhaseDampingChannel<float>::Create(0, 0, gamma);
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add channel.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("PD");

  // Set the args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["gamma"] = MakeArg(gamma);

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("");
  (*args_proto)["control_values"] = MakeControlArg("");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");

  NoisyQsimCircuit test_circuit;

  ASSERT_EQ(
      NoisyQsimCircuitFromProgram(program_proto, {}, 1, false, &test_circuit),
      tensorflow::Status::OK());
  AssertChannelEqual(test_circuit.channels[0], reference);
  ASSERT_EQ(test_circuit.channels.size(), 1);
  ASSERT_EQ(test_circuit.num_qubits, 1);
}

TEST(QsimCircuitParserTest, PhaseFlip) {
  float p = 0.1234;
  auto reference = qsim::Cirq::PhaseFlipChannel<float>::Create(0, 0, p);
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add channel.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("PF");

  // Set the args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["p"] = MakeArg(p);

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("");
  (*args_proto)["control_values"] = MakeControlArg("");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");

  NoisyQsimCircuit test_circuit;

  ASSERT_EQ(
      NoisyQsimCircuitFromProgram(program_proto, {}, 1, false, &test_circuit),
      tensorflow::Status::OK());
  AssertChannelEqual(test_circuit.channels[0], reference);
  ASSERT_EQ(test_circuit.channels.size(), 1);
  ASSERT_EQ(test_circuit.num_qubits, 1);
}

TEST(QsimCircuitParserTest, BitFlip) {
  float p = 0.1234;
  auto reference = qsim::Cirq::BitFlipChannel<float>::Create(0, 0, p);
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add channel.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("BF");

  // Set the args.
  google::protobuf::Map<std::string, Arg>* args_proto =
      operations_proto->mutable_args();
  (*args_proto)["p"] = MakeArg(p);

  // Set the control args.
  (*args_proto)["control_qubits"] = MakeControlArg("");
  (*args_proto)["control_values"] = MakeControlArg("");

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");

  NoisyQsimCircuit test_circuit;

  ASSERT_EQ(
      NoisyQsimCircuitFromProgram(program_proto, {}, 1, false, &test_circuit),
      tensorflow::Status::OK());
  AssertChannelEqual(test_circuit.channels[0], reference);
  ASSERT_EQ(test_circuit.channels.size(), 1);
  ASSERT_EQ(test_circuit.num_qubits, 1);
}

TEST(QsimCircuitParserTest, NoisyEmpty) {
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  (void)circuit_proto->add_moments();

  NoisyQsimCircuit test_circuit;
  ASSERT_EQ(
      NoisyQsimCircuitFromProgram(program_proto, {}, 0, false, &test_circuit),
      tensorflow::Status::OK());
  ASSERT_EQ(test_circuit.channels.size(), 0);
  ASSERT_EQ(test_circuit.num_qubits, 0);
}

TEST(QsimCircuitParserTest, NoisyBadProto) {
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
  Moment* moments_proto = circuit_proto->add_moments();

  // Add channel.
  Operation* operations_proto = moments_proto->add_operations();
  Gate* gate_proto = operations_proto->mutable_gate();
  gate_proto->set_id("ABCDEFG");

  NoisyQsimCircuit test_circuit;
  ASSERT_EQ(
      NoisyQsimCircuitFromProgram(program_proto, {}, 1, false, &test_circuit),
      tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                         "Could not parse channel id: ABCDEFG"));
}

TEST(QsimCircuitParserTest, CircuitFromPauliTermPauli) {
  tfq::proto::PauliTerm pauli_proto;
  // The created circuit should not depend on the coefficient
  pauli_proto.set_coefficient_real(3.14);
  tfq::proto::PauliQubitPair* pair_proto = pauli_proto.add_paulis();
  pair_proto->set_qubit_id("0");
  pair_proto->set_pauli_type("X");

  // Build the corresponding correct circuit
  auto reference = qsim::Cirq::XPowGate<float>::Create(0, 0, 1.0, 0.0);
  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  tensorflow::Status status;

  // Check conversion
  status =
      QsimCircuitFromPauliTerm(pauli_proto, 1, &test_circuit, &fused_circuit);
  ASSERT_EQ(status, tensorflow::Status::OK());
  ASSERT_EQ(test_circuit.num_qubits, 1);
  ASSERT_EQ(test_circuit.gates.size(), 1);
  AssertOneQubitEqual(test_circuit.gates[0], reference);
}

TEST(QsimCircuitParserTest, CircuitFromPauliTermEmpty) {
  tfq::proto::PauliTerm pauli_proto;
  tensorflow::Status status;
  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  status =
      QsimCircuitFromPauliTerm(pauli_proto, 0, &test_circuit, &fused_circuit);
  ASSERT_EQ(status, tensorflow::Status::OK());
  ASSERT_EQ(test_circuit.num_qubits, 0);
  ASSERT_EQ(test_circuit.gates.size(), 0);
}

TEST(QsimCircuitParserTest, ZBasisCircuitFromPauliTermPauliX) {
  tfq::proto::PauliTerm pauli_proto;
  // The created circuit should not depend on the coefficient
  pauli_proto.set_coefficient_real(3.14);
  tfq::proto::PauliQubitPair* pair_proto = pauli_proto.add_paulis();
  pair_proto->set_qubit_id("0");
  pair_proto->set_pauli_type("X");

  // Build the corresponding correct circuit
  auto reference = qsim::Cirq::YPowGate<float>::Create(0, 0, -0.5, 0.0);
  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  tensorflow::Status status;

  // Check conversion
  status = QsimZBasisCircuitFromPauliTerm(pauli_proto, 1, &test_circuit,
                                          &fused_circuit);
  ASSERT_EQ(status, tensorflow::Status::OK());
  ASSERT_EQ(test_circuit.num_qubits, 1);
  ASSERT_EQ(test_circuit.gates.size(), 1);
  AssertOneQubitEqual(test_circuit.gates[0], reference);
}

TEST(QsimCircuitParserTest, ZBasisCircuitFromPauliTermPauliY) {
  tfq::proto::PauliTerm pauli_proto;
  // The created circuit should not depend on the coefficient
  pauli_proto.set_coefficient_real(3.14);
  tfq::proto::PauliQubitPair* pair_proto = pauli_proto.add_paulis();
  pair_proto->set_qubit_id("0");
  pair_proto->set_pauli_type("Y");

  // Build the corresponding correct circuit
  auto reference = qsim::Cirq::XPowGate<float>::Create(0, 0, 0.5, 0.0);
  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  tensorflow::Status status;

  // Check conversion
  status = QsimZBasisCircuitFromPauliTerm(pauli_proto, 1, &test_circuit,
                                          &fused_circuit);
  ASSERT_EQ(status, tensorflow::Status::OK());
  ASSERT_EQ(test_circuit.num_qubits, 1);
  ASSERT_EQ(test_circuit.gates.size(), 1);
  AssertOneQubitEqual(test_circuit.gates[0], reference);
}

TEST(QsimCircuitParserTest, ZBasisCircuitFromPauliTermPauliZ) {
  tfq::proto::PauliTerm pauli_proto;
  // The created circuit should not depend on the coefficient
  pauli_proto.set_coefficient_real(3.14);
  tfq::proto::PauliQubitPair* pair_proto = pauli_proto.add_paulis();
  pair_proto->set_qubit_id("0");
  pair_proto->set_pauli_type("Z");

  // Build the corresponding correct circuit
  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  tensorflow::Status status;

  // Check conversion
  status = QsimZBasisCircuitFromPauliTerm(pauli_proto, 1, &test_circuit,
                                          &fused_circuit);
  ASSERT_EQ(status, tensorflow::Status::OK());
  ASSERT_EQ(test_circuit.num_qubits, 1);
  ASSERT_EQ(test_circuit.gates.size(), 0);
}

TEST(QsimCircuitParserTest, ZBasisCircuitFromPauliTermPauliCompound) {
  tfq::proto::PauliTerm pauli_proto;
  // The created circuit should not depend on the coefficient
  pauli_proto.set_coefficient_real(3.14);
  tfq::proto::PauliQubitPair* pair_proto = pauli_proto.add_paulis();
  pair_proto->set_qubit_id("0");
  pair_proto->set_pauli_type("X");
  pair_proto = pauli_proto.add_paulis();
  pair_proto->set_qubit_id("1");
  pair_proto->set_pauli_type("Y");

  auto reference1 = qsim::Cirq::YPowGate<float>::Create(0, 1, -0.5, 0.0);
  auto reference2 = qsim::Cirq::XPowGate<float>::Create(0, 0, 0.5, 0.0);

  // Build the corresponding correct circuit
  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  tensorflow::Status status;

  // Check conversion
  status = QsimZBasisCircuitFromPauliTerm(pauli_proto, 2, &test_circuit,
                                          &fused_circuit);
  ASSERT_EQ(status, tensorflow::Status::OK());
  ASSERT_EQ(test_circuit.num_qubits, 2);
  ASSERT_EQ(test_circuit.gates.size(), 2);
  AssertOneQubitEqual(test_circuit.gates[0], reference1);
  AssertOneQubitEqual(test_circuit.gates[1], reference2);
}

TEST(QsimCircuitParserTest, ZBasisCircuitFromPauliTermEmpty) {
  tfq::proto::PauliTerm pauli_proto;
  tensorflow::Status status;
  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  status = QsimZBasisCircuitFromPauliTerm(pauli_proto, 0, &test_circuit,
                                          &fused_circuit);
  ASSERT_EQ(status, tensorflow::Status::OK());
  ASSERT_EQ(test_circuit.num_qubits, 0);
  ASSERT_EQ(test_circuit.gates.size(), 0);
}

}  // namespace
}  // namespace tfq
