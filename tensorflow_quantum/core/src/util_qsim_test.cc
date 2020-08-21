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

#include "tensorflow_quantum/core/src/util_qsim.h"

#include <vector>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/formux.h"
#include "../qsim/lib/fuser.h"
#include "../qsim/lib/fuser_basic.h"
#include "../qsim/lib/gate_appl.h"
#include "../qsim/lib/gates_cirq.h"
#include "../qsim/lib/io.h"
#include "../qsim/lib/simmux.h"
#include "absl/container/flat_hash_map.h"
#include "gtest/gtest.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"

namespace tfq {
namespace {

using ::tensorflow::Status;
using ::tfq::proto::PauliQubitPair;
using ::tfq::proto::PauliSum;
using ::tfq::proto::PauliTerm;

typedef absl::flat_hash_map<std::string, std::pair<int, float>> SymbolMap;
typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;

class TwoTermSampledExpectationFixture
    : public ::testing::TestWithParam<std::tuple<std::string, float>> {};

TEST_P(TwoTermSampledExpectationFixture, CorrectnessTest) {
  // Create circuit to prepare initial state.
  QsimCircuit simple_circuit;
  simple_circuit.num_qubits = 2;
  simple_circuit.gates.push_back(
      qsim::Cirq::XPowGate<float>::Create(0, 1, 0.25, 0.0));
  simple_circuit.gates.push_back(
      qsim::Cirq::CXPowGate<float>::Create(1, 1, 0, 1.0, 0.0));
  simple_circuit.gates.push_back(
      qsim::Cirq::YPowGate<float>::Create(2, 0, 0.5, 0.0));

  auto fused_circuit = qsim::BasicGateFuser<qsim::IO, QsimGate>().FuseGates(
      simple_circuit.num_qubits, simple_circuit.gates);

  // Instantiate qsim objects.
  qsim::Simulator<qsim::SequentialFor> sim(2, 1);
  qsim::Simulator<qsim::SequentialFor>::StateSpace ss(2, 1);
  auto sv = ss.CreateState();
  ss.SetStateZero(sv);
  auto scratch = ss.CreateState();

  // Prepare initial state.
  ss.SetStateZero(sv);
  for (int j = 0; j < fused_circuit.size(); j++) {
    qsim::ApplyFusedGate(sim, fused_circuit[j], sv);
  }

  PauliSum p_sum;
  std::string p_string = std::get<0>(GetParam());

  // Initialize pauli sum.
  PauliTerm* p_term_scratch = p_sum.add_terms();
  p_term_scratch->set_coefficient_real(0.1234);
  PauliQubitPair* pair_proto = p_term_scratch->add_paulis();
  pair_proto->set_qubit_id(std::to_string(0));
  pair_proto->set_pauli_type(p_string.substr(0, 1));
  pair_proto = p_term_scratch->add_paulis();
  pair_proto->set_qubit_id(std::to_string(1));
  pair_proto->set_pauli_type(p_string.substr(1, 1));

  // Compute expectation and compare to reference values.
  float exp_v = 0;
  Status s = tfq::ComputeSampledExpectationQsim(p_sum, sim, ss, sv, scratch,
                                                1000000, &exp_v);

  EXPECT_NEAR(exp_v, std::get<1>(GetParam()), 1e-3);
}

// clang-format off
INSTANTIATE_TEST_CASE_P(
    SingleTermSampledExpectationTests, TwoTermSampledExpectationFixture,
    ::testing::Values(std::make_tuple("ZZ", 0.0),
                      std::make_tuple("ZX", 0.1234),
                      std::make_tuple("ZY", 0.0),
                      std::make_tuple("XZ", 0.0),
                      std::make_tuple("XX", 0.0),
                      std::make_tuple("XY", -0.08725),
                      std::make_tuple("YZ", 0.08725),
                      std::make_tuple("YX", 0.0),
                      std::make_tuple("YY", 0.0)));
// clang-format on

class TwoTermExpectationFixture
    : public ::testing::TestWithParam<std::tuple<std::string, float>> {};

TEST_P(TwoTermExpectationFixture, CorrectnessTest) {
  // Create circuit to prepare initial state.
  QsimCircuit simple_circuit;
  simple_circuit.num_qubits = 2;
  simple_circuit.gates.push_back(
      qsim::Cirq::XPowGate<float>::Create(0, 1, 0.25, 0.0));
  simple_circuit.gates.push_back(
      qsim::Cirq::CXPowGate<float>::Create(1, 1, 0, 1.0, 0.0));
  simple_circuit.gates.push_back(
      qsim::Cirq::YPowGate<float>::Create(2, 0, 0.5, 0.0));

  auto fused_circuit = qsim::BasicGateFuser<qsim::IO, QsimGate>().FuseGates(
      simple_circuit.num_qubits, simple_circuit.gates);

  // Instantiate qsim objects.
  qsim::Simulator<qsim::SequentialFor> sim(2, 1);
  qsim::Simulator<qsim::SequentialFor>::StateSpace ss(2, 1);
  auto sv = ss.CreateState();
  ss.SetStateZero(sv);
  auto scratch = ss.CreateState();

  // Prepare initial state.
  ss.SetStateZero(sv);
  for (int j = 0; j < fused_circuit.size(); j++) {
    qsim::ApplyFusedGate(sim, fused_circuit[j], sv);
  }

  PauliSum p_sum;
  std::string p_string = std::get<0>(GetParam());

  // Initialize pauli sum.
  PauliTerm* p_term_scratch = p_sum.add_terms();
  p_term_scratch->set_coefficient_real(0.1234);
  PauliQubitPair* pair_proto = p_term_scratch->add_paulis();
  pair_proto->set_qubit_id(std::to_string(0));
  pair_proto->set_pauli_type(p_string.substr(0, 1));
  pair_proto = p_term_scratch->add_paulis();
  pair_proto->set_qubit_id(std::to_string(1));
  pair_proto->set_pauli_type(p_string.substr(1, 1));

  // Compute expectation and compare to reference values.
  float exp_v = 0;
  Status s = tfq::ComputeExpectationQsim(p_sum, sim, ss, sv, scratch, &exp_v);

  EXPECT_NEAR(exp_v, std::get<1>(GetParam()), 1e-5);
}

// clang-format off
INSTANTIATE_TEST_CASE_P(
    SingleTermExpectationTests, TwoTermExpectationFixture,
    ::testing::Values(std::make_tuple("ZZ", 0.0),
                      std::make_tuple("ZX", 0.1234),
                      std::make_tuple("ZY", 0.0),
                      std::make_tuple("XZ", 0.0),
                      std::make_tuple("XX", 0.0),
                      std::make_tuple("XY", -0.08725),
                      std::make_tuple("YZ", 0.08725),
                      std::make_tuple("YX", 0.0),
                      std::make_tuple("YY", 0.0)));
// clang-format on

TEST(UtilQsimTest, SampledEmptyTermCase) {
  // test that the identity term gets picked up correctly as an empty
  // pauliTerm.
  // Instantiate qsim objects.
  qsim::Simulator<qsim::SequentialFor> sim(2, 1);
  qsim::Simulator<qsim::SequentialFor>::StateSpace ss(2, 1);
  auto sv = ss.CreateState();
  ss.SetStateZero(sv);
  auto scratch = ss.CreateState();

  PauliSum p_sum_empty;

  // Initialize pauli sum.
  PauliTerm* p_term_empty = p_sum_empty.add_terms();
  p_term_empty->set_coefficient_real(0.1234);

  // Compute expectation and compare to reference values.
  float exp_v = 0;
  Status s = tfq::ComputeSampledExpectationQsim(p_sum_empty, sim, ss, sv,
                                                scratch, 100, &exp_v);

  EXPECT_NEAR(exp_v, 0.1234, 1e-5);
}

TEST(UtilQsimTest, EmptyTermCase) {
  // test that the identity term gets picked up correctly as an empty
  // pauliTerm.
  // Instantiate qsim objects.
  qsim::Simulator<qsim::SequentialFor> sim(2, 1);
  qsim::Simulator<qsim::SequentialFor>::StateSpace ss(2, 1);
  auto sv = ss.CreateState();
  ss.SetStateZero(sv);
  auto scratch = ss.CreateState();

  PauliSum p_sum_empty;

  // Initialize pauli sum.
  PauliTerm* p_term_empty = p_sum_empty.add_terms();
  p_term_empty->set_coefficient_real(0.1234);

  // Compute expectation and compare to reference values.
  float exp_v = 0;
  Status s =
      tfq::ComputeExpectationQsim(p_sum_empty, sim, ss, sv, scratch, &exp_v);

  EXPECT_NEAR(exp_v, 0.1234, 1e-5);
}

TEST(UtilQsimTest, SampledCompoundCase) {
  // Create circuit to prepare initial state.
  QsimCircuit simple_circuit;
  simple_circuit.num_qubits = 2;
  simple_circuit.gates.push_back(
      qsim::Cirq::XPowGate<float>::Create(0, 1, 0.25, 0.0));
  simple_circuit.gates.push_back(
      qsim::Cirq::CXPowGate<float>::Create(1, 1, 0, 1.0, 0.0));
  simple_circuit.gates.push_back(
      qsim::Cirq::YPowGate<float>::Create(2, 0, 0.5, 0.0));

  auto fused_circuit = qsim::BasicGateFuser<qsim::IO, QsimGate>().FuseGates(
      simple_circuit.num_qubits, simple_circuit.gates);

  // Instantiate qsim objects.
  qsim::Simulator<qsim::SequentialFor> sim(2, 1);
  qsim::Simulator<qsim::SequentialFor>::StateSpace ss(2, 1);
  auto sv = ss.CreateState();
  ss.SetStateZero(sv);
  auto scratch = ss.CreateState();

  // Prepare initial state.
  ss.SetStateZero(sv);
  for (int j = 0; j < fused_circuit.size(); j++) {
    qsim::ApplyFusedGate(sim, fused_circuit[j], sv);
  }

  PauliSum p_sum;

  // Initialize pauli sum.
  // 0.1234 ZX
  PauliTerm* p_term_scratch = p_sum.add_terms();
  p_term_scratch->set_coefficient_real(0.1234);
  PauliQubitPair* pair_proto = p_term_scratch->add_paulis();
  pair_proto->set_qubit_id(std::to_string(0));
  pair_proto->set_pauli_type("Z");
  pair_proto = p_term_scratch->add_paulis();
  pair_proto->set_qubit_id(std::to_string(1));
  pair_proto->set_pauli_type("X");

  // -3.0 X
  p_term_scratch = p_sum.add_terms();
  p_term_scratch->set_coefficient_real(-3.0);
  pair_proto = p_term_scratch->add_paulis();
  pair_proto->set_qubit_id(std::to_string(0));
  pair_proto->set_pauli_type("X");

  // 4.0 I
  p_term_scratch = p_sum.add_terms();
  p_term_scratch->set_coefficient_real(4.0);
  // Compute expectation and compare to reference values.
  float exp_v = 0;
  Status s = tfq::ComputeSampledExpectationQsim(p_sum, sim, ss, sv, scratch,
                                                10000000, &exp_v);

  EXPECT_NEAR(exp_v, 4.1234, 1e-3);
}

TEST(UtilQsimTest, CompoundCase) {
  // Create circuit to prepare initial state.
  QsimCircuit simple_circuit;
  simple_circuit.num_qubits = 2;
  simple_circuit.gates.push_back(
      qsim::Cirq::XPowGate<float>::Create(0, 1, 0.25, 0.0));
  simple_circuit.gates.push_back(
      qsim::Cirq::CXPowGate<float>::Create(1, 1, 0, 1.0, 0.0));
  simple_circuit.gates.push_back(
      qsim::Cirq::YPowGate<float>::Create(2, 0, 0.5, 0.0));

  auto fused_circuit = qsim::BasicGateFuser<qsim::IO, QsimGate>().FuseGates(
      simple_circuit.num_qubits, simple_circuit.gates);

  // Instantiate qsim objects.
  qsim::Simulator<qsim::SequentialFor> sim(2, 1);
  qsim::Simulator<qsim::SequentialFor>::StateSpace ss(2, 1);
  auto sv = ss.CreateState();
  ss.SetStateZero(sv);
  auto scratch = ss.CreateState();

  // Prepare initial state.
  ss.SetStateZero(sv);
  for (int j = 0; j < fused_circuit.size(); j++) {
    qsim::ApplyFusedGate(sim, fused_circuit[j], sv);
  }

  PauliSum p_sum;

  // Initialize pauli sum.
  // 0.1234 ZX
  PauliTerm* p_term_scratch = p_sum.add_terms();
  p_term_scratch->set_coefficient_real(0.1234);
  PauliQubitPair* pair_proto = p_term_scratch->add_paulis();
  pair_proto->set_qubit_id(std::to_string(0));
  pair_proto->set_pauli_type("Z");
  pair_proto = p_term_scratch->add_paulis();
  pair_proto->set_qubit_id(std::to_string(1));
  pair_proto->set_pauli_type("X");

  // -3.0 X
  p_term_scratch = p_sum.add_terms();
  p_term_scratch->set_coefficient_real(-3.0);
  pair_proto = p_term_scratch->add_paulis();
  pair_proto->set_qubit_id(std::to_string(0));
  pair_proto->set_pauli_type("X");

  // 4.0 I
  p_term_scratch = p_sum.add_terms();
  p_term_scratch->set_coefficient_real(4.0);
  // Compute expectation and compare to reference values.
  float exp_v = 0;
  Status s = tfq::ComputeExpectationQsim(p_sum, sim, ss, sv, scratch, &exp_v);

  EXPECT_NEAR(exp_v, 4.1234, 1e-5);
}

TEST(UtilQsimTest, ApplyGateDagger) {
  // Create circuit to prepare initial state.
  QsimCircuit simple_circuit;
  simple_circuit.num_qubits = 2;
  simple_circuit.gates.push_back(
      qsim::Cirq::XPowGate<float>::Create(0, 1, 0.25, 0.0));
  simple_circuit.gates.push_back(
      qsim::Cirq::CXPowGate<float>::Create(1, 1, 0, 1.0, 0.0));
  simple_circuit.gates.push_back(
      qsim::Cirq::YPowGate<float>::Create(2, 0, 0.5, 0.0));
  // Instantiate qsim objects.
  qsim::Simulator<qsim::SequentialFor> sim(2, 1);
  qsim::Simulator<qsim::SequentialFor>::StateSpace ss(2, 1);
  auto sv = ss.CreateState();
  ss.SetStateZero(sv);
  auto scratch = ss.CreateState();

  // Prepare initial state.
  ss.SetStateZero(sv);
  for (int i = 0; i < simple_circuit.gates.size(); i++) {
    qsim::ApplyGate(sim, simple_circuit.gates[i], sv);
  }
  for (int i = simple_circuit.gates.size() - 1; i >= 0; i--) {
    ApplyGateDagger(sim, simple_circuit.gates[i], sv);
  }

  // Test that we reverted back to zero.
  EXPECT_NEAR(ss.GetAmpl(sv, 0).real(), 1.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 0).imag(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 1).real(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 1).imag(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 2).real(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 2).imag(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 3).real(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 3).imag(), 0.0, 1e-5);
}

TEST(UtilQsimTest, ApplyFusedGateDagger) {
  // Create circuit to prepare initial state.
  QsimCircuit simple_circuit;
  simple_circuit.num_qubits = 2;
  simple_circuit.gates.push_back(
      qsim::Cirq::XPowGate<float>::Create(0, 1, 0.25, 0.0));
  simple_circuit.gates.push_back(
      qsim::Cirq::CXPowGate<float>::Create(1, 1, 0, 1.0, 0.0));
  simple_circuit.gates.push_back(
      qsim::Cirq::YPowGate<float>::Create(2, 0, 0.5, 0.0));
  // Instantiate qsim objects.
  qsim::Simulator<qsim::SequentialFor> sim(2, 1);
  qsim::Simulator<qsim::SequentialFor>::StateSpace ss(2, 1);
  auto sv = ss.CreateState();
  ss.SetStateZero(sv);
  auto scratch = ss.CreateState();

  // Prepare initial state.
  ss.SetStateZero(sv);
  auto fused_circuit = qsim::BasicGateFuser<qsim::IO, QsimGate>().FuseGates(
      simple_circuit.num_qubits, simple_circuit.gates);

  // Prepare initial state.
  ss.SetStateZero(sv);
  for (int j = 0; j < fused_circuit.size(); j++) {
    qsim::ApplyFusedGate(sim, fused_circuit[j], sv);
  }
  for (int i = fused_circuit.size() - 1; i >= 0; i--) {
    ApplyFusedGateDagger(sim, fused_circuit[i], sv);
  }

  // Test that we reverted back to zero.
  EXPECT_NEAR(ss.GetAmpl(sv, 0).real(), 1.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 0).imag(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 1).real(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 1).imag(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 2).real(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 2).imag(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 3).real(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 3).imag(), 0.0, 1e-5);
}

TEST(UtilQsimTest, AccumulateOperatorsBasic) {
  // Create circuit to prepare initial state.
  QsimCircuit simple_circuit;
  simple_circuit.num_qubits = 2;
  simple_circuit.gates.push_back(
      qsim::Cirq::XPowGate<float>::Create(0, 1, 0.25, 0.0));
  simple_circuit.gates.push_back(
      qsim::Cirq::CXPowGate<float>::Create(1, 1, 0, 1.0, 0.0));
  simple_circuit.gates.push_back(
      qsim::Cirq::YPowGate<float>::Create(2, 0, 0.5, 0.0));

  auto fused_circuit = qsim::BasicGateFuser<qsim::IO, QsimGate>().FuseGates(
      simple_circuit.num_qubits, simple_circuit.gates);

  // Instantiate qsim objects.
  qsim::Simulator<qsim::SequentialFor> sim(2, 1);
  qsim::Simulator<qsim::SequentialFor>::StateSpace ss(2, 1);
  auto sv = ss.CreateState();
  ss.SetStateZero(sv);
  auto scratch = ss.CreateState();
  auto dest = ss.CreateState();

  // Prepare initial state.
  ss.SetStateZero(sv);
  for (int j = 0; j < fused_circuit.size(); j++) {
    qsim::ApplyFusedGate(sim, fused_circuit[j], sv);
  }

  PauliSum p_sum;

  // Initialize pauli sum.
  // 0.1234 ZX
  PauliTerm* p_term_scratch = p_sum.add_terms();
  p_term_scratch->set_coefficient_real(0.1234);
  PauliQubitPair* pair_proto = p_term_scratch->add_paulis();
  pair_proto->set_qubit_id(std::to_string(0));
  pair_proto->set_pauli_type("Z");
  pair_proto = p_term_scratch->add_paulis();
  pair_proto->set_qubit_id(std::to_string(1));
  pair_proto->set_pauli_type("X");

  // -3.0 X
  p_term_scratch = p_sum.add_terms();
  p_term_scratch->set_coefficient_real(-3.0);
  pair_proto = p_term_scratch->add_paulis();
  pair_proto->set_qubit_id(std::to_string(0));
  pair_proto->set_pauli_type("X");

  // 4.0 I
  p_term_scratch = p_sum.add_terms();
  p_term_scratch->set_coefficient_real(4.0);

  // A second sum.
  PauliSum p_sum2;
  PauliTerm* p_term_scratch2 = p_sum2.add_terms();
  p_term_scratch2->set_coefficient_real(-5.0);

  // 0.5 * (0.123ZX -3X + 4I) + 0.25 * (-5I) applied onto psi.
  AccumulateOperators({p_sum, p_sum2}, {0.5, 0.25}, sim, ss, sv, scratch, dest);

  // Check that dest got accumulated onto.
  EXPECT_NEAR(ss.GetAmpl(dest, 0).real(), 0.577925, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(dest, 0).imag(), 0.334574, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(dest, 1).real(), -0.172075, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(dest, 1).imag(), 0.645234, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(dest, 2).real(), -0.577925, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(dest, 2).imag(), -0.821275, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(dest, 3).real(), -0.172075, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(dest, 3).imag(), -0.989384, 1e-5);

  // Check that sv remains unchanged.
  EXPECT_NEAR(ss.GetAmpl(sv, 0).real(), 0.25, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 0).imag(), 0.60355, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 1).real(), 0.25, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 1).imag(), 0.60355, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 2).real(), -0.25, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 2).imag(), 0.10355, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 3).real(), 0.25, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 3).imag(), -0.10355, 1e-5);

  // Check that as a consequence of running scratch is a copy of sv.
  EXPECT_NEAR(ss.GetAmpl(scratch, 0).real(), 0.25, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(scratch, 0).imag(), 0.60355, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(scratch, 1).real(), 0.25, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(scratch, 1).imag(), 0.60355, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(scratch, 2).real(), -0.25, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(scratch, 2).imag(), 0.10355, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(scratch, 3).real(), 0.25, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(scratch, 3).imag(), -0.10355, 1e-5);
}

TEST(UtilQsimTest, AccumulateOperatorsEmpty) {
  // Instantiate qsim objects.
  qsim::Simulator<qsim::SequentialFor> sim(2, 1);
  qsim::Simulator<qsim::SequentialFor>::StateSpace ss(2, 1);
  auto sv = ss.CreateState();
  ss.SetStateZero(sv);
  auto scratch = ss.CreateState();
  auto dest = ss.CreateState();

  AccumulateOperators({}, {}, sim, ss, sv, scratch, dest);

  // Check sv is still in zero state.
  EXPECT_NEAR(ss.GetAmpl(sv, 0).real(), 1.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 0).imag(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 1).real(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 1).imag(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 2).real(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 2).imag(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 3).real(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(sv, 3).imag(), 0.0, 1e-5);

  // Check scratch is a copy of sv.
  EXPECT_NEAR(ss.GetAmpl(scratch, 0).real(), 1.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(scratch, 0).imag(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(scratch, 1).real(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(scratch, 1).imag(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(scratch, 2).real(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(scratch, 2).imag(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(scratch, 3).real(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(scratch, 3).imag(), 0.0, 1e-5);

  // Check that dest contains all zeros.
  EXPECT_NEAR(ss.GetAmpl(dest, 0).real(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(dest, 0).imag(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(dest, 1).real(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(dest, 1).imag(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(dest, 2).real(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(dest, 2).imag(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(dest, 3).real(), 0.0, 1e-5);
  EXPECT_NEAR(ss.GetAmpl(scratch, 3).imag(), 0.0, 1e-5);
}

}  // namespace
}  // namespace tfq
