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

using ::cirq::google::api::v2::Arg;
using ::cirq::google::api::v2::Circuit;
using ::cirq::google::api::v2::Gate;
using ::cirq::google::api::v2::Moment;
using ::cirq::google::api::v2::Operation;
using ::cirq::google::api::v2::Program;
using ::cirq::google::api::v2::Qubit;

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

inline void AssertTwoQubitEqual(const QsimGate& a, const QsimGate& b) {
  for (int i = 0; i < 32; i++) {
    ASSERT_NEAR(a.matrix[i], b.matrix[i], 1e-5);
  }
  ASSERT_EQ(a.qubits[0], b.qubits[0]);
  ASSERT_EQ(a.qubits[1], b.qubits[1]);
}

inline void AssertOneQubitEqual(const QsimGate& a, const QsimGate& b) {
  for (int i = 0; i < 8; i++) {
    ASSERT_NEAR(a.matrix[i], b.matrix[i], 1e-5);
  }
  ASSERT_EQ(a.qubits[0], b.qubits[0]);
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

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");
  qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("1");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap symbol_map = {{"placeholder", std::pair<int, float>(1, 2 * exp)}};

  // Test case where we have a placeholder.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status::OK());
  AssertTwoQubitEqual(test_circuit.gates[0], ref_gate);

  test_circuit.gates.clear();
  fused_circuit.clear();
  (*args_proto)["exponent"] = MakeArg(exp);
  (*args_proto)["exponent_scalar"] = MakeArg(1.0);
  symbol_map.clear();

  // Test case where we have all float values.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status::OK());
  AssertTwoQubitEqual(test_circuit.gates[0], ref_gate);

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

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap symbol_map = {{"placeholder", std::pair<int, float>(1, 2 * exp)}};

  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 1, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status::OK());
  AssertOneQubitEqual(test_circuit.gates[0], ref_gate);

  test_circuit.gates.clear();
  fused_circuit.clear();
  (*args_proto)["exponent"] = MakeArg(exp);
  (*args_proto)["exponent_scalar"] = MakeArg(1.0);
  symbol_map.clear();

  // Test case where we have all float values.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 1, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status::OK());
  AssertOneQubitEqual(test_circuit.gates[0], ref_gate);

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

INSTANTIATE_TEST_CASE_P(
    SingleQubitEigenTests, SingleQubitEigenFixture,
    ::testing::Values(
        std::make_tuple("HP", &qsim::Cirq::HPowGate<float>::Create),
        std::make_tuple("XP", &qsim::Cirq::XPowGate<float>::Create),
        std::make_tuple("YP", &qsim::Cirq::YPowGate<float>::Create),
        std::make_tuple("ZP", &qsim::Cirq::ZPowGate<float>::Create)));

TEST(QsimCircuitParserTest, SingleConstantGate) {
  absl::flat_hash_map<std::string, QsimGate> reference = {
      {"I", qsim::Cirq::I<float>::Create(0, 0)}};
  for (auto kv : reference) {
    Program program_proto;
    Circuit* circuit_proto = program_proto.mutable_circuit();
    circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);
    Moment* moments_proto = circuit_proto->add_moments();

    // Add gate.
    Operation* operations_proto = moments_proto->add_operations();
    Gate* gate_proto = operations_proto->mutable_gate();
    gate_proto->set_id(kv.first);

    // Set the qubits.
    Qubit* qubits_proto = operations_proto->add_qubits();
    qubits_proto->set_id("0");

    QsimCircuit test_circuit;
    std::vector<qsim::GateFused<QsimGate>> fused_circuit;
    SymbolMap empty_map;

    ASSERT_EQ(QsimCircuitFromProgram(program_proto, empty_map, 1, &test_circuit,
                                     &fused_circuit),
              tensorflow::Status::OK());
    AssertOneQubitEqual(test_circuit.gates[0], kv.second);
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

    // Set the qubits.
    Qubit* qubits_proto = operations_proto->add_qubits();
    qubits_proto->set_id("0");
    qubits_proto = operations_proto->add_qubits();
    qubits_proto->set_id("1");

    QsimCircuit test_circuit;
    std::vector<qsim::GateFused<QsimGate>> fused_circuit;
    SymbolMap empty_map;

    ASSERT_EQ(QsimCircuitFromProgram(program_proto, empty_map, 2, &test_circuit,
                                     &fused_circuit),
              tensorflow::Status::OK());
    AssertTwoQubitEqual(test_circuit.gates[0], kv.second);
  }
}

TEST(QsimCircuitParserTest, FsimGateTest) {
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

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");
  qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("1");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap symbol_map = {{"alpha", std::pair<int, float>(0, 2 * theta)},
                          {"beta", std::pair<int, float>(1, 5 * phi)}};

  // Test symbol resolution.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status::OK());
  AssertTwoQubitEqual(test_circuit.gates[0], reference);

  symbol_map.clear();
  test_circuit.gates.clear();
  fused_circuit.clear();
  (*args_proto)["theta"] = MakeArg(theta);
  (*args_proto)["theta_scalar"] = MakeArg(1.0);
  (*args_proto)["phi"] = MakeArg(phi);
  (*args_proto)["phi_scalar"] = MakeArg(1.0);

  // Test float values only.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status::OK());
  AssertTwoQubitEqual(test_circuit.gates[0], reference);

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

TEST(QsimCircuitParserTest, PhasedISwapTest) {
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

  // Test symbol resolution.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status::OK());
  AssertTwoQubitEqual(test_circuit.gates[0], reference);

  symbol_map.clear();
  test_circuit.gates.clear();
  fused_circuit.clear();
  (*args_proto)["phase_exponent"] = MakeArg(phase_exponent);
  (*args_proto)["phase_exponent_scalar"] = MakeArg(1.0);
  (*args_proto)["exponent"] = MakeArg(exponent);
  (*args_proto)["exponent_scalar"] = MakeArg(1.0);

  // Test float values only.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 2, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status::OK());
  AssertTwoQubitEqual(test_circuit.gates[0], reference);

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

TEST(QsimCircuitParserTest, PhasedXPowTest) {
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

  // Set the qubits.
  Qubit* qubits_proto = operations_proto->add_qubits();
  qubits_proto->set_id("0");

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap symbol_map = {
      {"alpha", std::pair<int, float>(0, 2 * phase_exponent)},
      {"beta", std::pair<int, float>(1, 5 * exponent)}};

  // Test symbol resolution.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 1, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status::OK());
  AssertOneQubitEqual(test_circuit.gates[0], reference);

  symbol_map.clear();
  test_circuit.gates.clear();
  fused_circuit.clear();
  (*args_proto)["phase_exponent"] = MakeArg(phase_exponent);
  (*args_proto)["phase_exponent_scalar"] = MakeArg(1.0);
  (*args_proto)["exponent"] = MakeArg(exponent);
  (*args_proto)["exponent_scalar"] = MakeArg(1.0);
  (*args_proto)["global_shift"] = MakeArg(gs);

  // Test float values only.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, symbol_map, 1, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status::OK());
  AssertOneQubitEqual(test_circuit.gates[0], reference);

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

TEST(QsimCircuitParserTest, EmptyTest) {
  Program program_proto;
  Circuit* circuit_proto = program_proto.mutable_circuit();
  circuit_proto->set_scheduling_strategy(circuit_proto->MOMENT_BY_MOMENT);

  QsimCircuit test_circuit;
  std::vector<qsim::GateFused<QsimGate>> fused_circuit;
  SymbolMap empty_map;

  // Ensure that nothing bad happens with an empty circuit.
  ASSERT_EQ(QsimCircuitFromProgram(program_proto, empty_map, 2, &test_circuit,
                                   &fused_circuit),
            tensorflow::Status::OK());
  ASSERT_EQ(test_circuit.gates.size(), 0);
  ASSERT_EQ(fused_circuit.size(), 0);
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

}  // namespace
}  // namespace tfq
