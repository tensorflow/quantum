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

#include "tensorflow_quantum/core/qsim/state_space.h"

#include <cmath>
#include <complex>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/qsim/mux.h"
#include "tensorflow_quantum/core/src/circuit.h"

namespace tfq {
namespace qsim {
namespace {

TEST(StateSpaceTest, Initialization) {
  uint64_t num_qubits = 4;
  uint64_t num_threads = 5;
  auto state =
      std::unique_ptr<StateSpace>(GetStateSpace(num_qubits, num_threads));
  ASSERT_FALSE(state->Valid());
  ASSERT_FALSE(state->GetRawState());
  ASSERT_EQ(state->GetDimension(), 1 << num_qubits);
  ASSERT_EQ(state->GetNumQubits(), num_qubits);
  ASSERT_EQ(state->GetNumThreads(), num_threads);

  state->CreateState();
  ASSERT_TRUE(state->Valid());
  ASSERT_TRUE(state->GetRawState());
  ASSERT_EQ(state->GetDimension(), 1 << num_qubits);
  ASSERT_EQ(state->GetNumQubits(), num_qubits);
  ASSERT_EQ(state->GetNumThreads(), num_threads);

#ifdef __AVX2__
  tfq::qsim::StateSpaceType state_space_type = tfq::qsim::StateSpaceType::AVX;
#elif __SSE4_1__
  tfq::qsim::StateSpaceType state_space_type = tfq::qsim::StateSpaceType::SSE;
#else
  tfq::qsim::StateSpaceType state_space_type = tfq::qsim::StateSpaceType::SLOW;
#endif
  ASSERT_EQ(state->GetType(), state_space_type);

  state->DeleteState();
  ASSERT_FALSE(state->Valid());
  ASSERT_FALSE(state->GetRawState());
}

TEST(StateSpaceTest, CloneTest) {
  auto state = std::unique_ptr<StateSpace>(GetStateSpace(5, 3));
  auto state_clone = std::unique_ptr<StateSpace>(state->Clone());

  ASSERT_EQ(state->GetDimension(), state_clone->GetDimension());
  ASSERT_EQ(state->GetNumQubits(), state_clone->GetNumQubits());
  ASSERT_EQ(state->GetNumThreads(), state_clone->GetNumThreads());
}

TEST(StateSpaceTest, Amplitudes) {
  auto state = std::unique_ptr<StateSpace>(GetStateSpace(2, 1));
  state->CreateState();

  std::complex<float> ampl_00(0.1, 0.5);
  std::complex<float> ampl_01(0.2, 0.6);
  std::complex<float> ampl_10(0.3, 0.7);
  std::complex<float> ampl_11(0.4, 0.8);

  state->SetAmpl(0, ampl_00);
  state->SetAmpl(1, ampl_01);
  state->SetAmpl(2, ampl_10);
  state->SetAmpl(3, ampl_11);

  ASSERT_EQ(state->GetAmpl(0), ampl_00);
  ASSERT_EQ(state->GetAmpl(1), ampl_01);
  ASSERT_EQ(state->GetAmpl(2), ampl_10);
  ASSERT_EQ(state->GetAmpl(3), ampl_11);

  state->SetStateZero();
  ASSERT_EQ(state->GetAmpl(0), std::complex<float>(1, 0));
  ASSERT_EQ(state->GetAmpl(1), std::complex<float>(0, 0));
  ASSERT_EQ(state->GetAmpl(2), std::complex<float>(0, 0));
  ASSERT_EQ(state->GetAmpl(3), std::complex<float>(0, 0));
}

TEST(StateSpaceTest, CopyFromGetRealInnerProduct) {
  auto state = std::unique_ptr<StateSpace>(GetStateSpace(12, 1));
  state->CreateState();
  for (uint64_t i = 0; i < state->GetDimension(); i++) {
    state->SetAmpl(i, std::complex<float>(std::sqrt(i), std::sqrt(i)));
  }

  auto state_clone = std::unique_ptr<StateSpace>(state->Clone());
  state_clone->CreateState();
  state_clone->CopyFrom(*state);

  for (uint64_t i = 0; i < state->GetDimension(); i++) {
    ASSERT_EQ(state->GetAmpl(i), state_clone->GetAmpl(i));
  }

  double real_inner_product =
      state->GetDimension() * state->GetDimension() - state->GetDimension();

  EXPECT_NEAR(state->GetRealInnerProduct(*state_clone), real_inner_product,
              1E-2);
}

TEST(StateSpaceTest, ApplyGate1) {
  auto state = std::unique_ptr<StateSpace>(GetStateSpace(5, 1));
  state->CreateState();
  const float matrix_h[] = {1.0 / std::sqrt(2), 0.0, 1.0 / std::sqrt(2),  0.0,
                            1.0 / std::sqrt(2), 0.0, -1.0 / std::sqrt(2), 0.0};
  state->SetStateZero();
  state->SetAmpl(0, std::complex<float>(0.0, 0.0));
  state->SetAmpl(1, std::complex<float>(1.0, 0.0));
  switch (state->GetType()) {
    case StateSpaceType::AVX:
      ASSERT_EQ(
          state->ApplyGate1(matrix_h),
          tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                             "AVX simulator doesn't support small circuits."));
      break;
    case StateSpaceType::SLOW:
      state->ApplyGate1(matrix_h);
      ASSERT_EQ(state->GetAmpl(0), std::complex<float>(1 / std::sqrt(2), 0.0));
      ASSERT_EQ(state->GetAmpl(1), std::complex<float>(-1 / std::sqrt(2), 0.0));
      break;
    case StateSpaceType::SSE:
      ASSERT_EQ(
          state->ApplyGate1(matrix_h),
          tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                             "SSE simulator doesn't support small circuits."));
      break;
  }
}

TEST(StateSpaceTest, ApplyGate2) {
  auto state = std::unique_ptr<StateSpace>(GetStateSpace(6, 1));
  state->CreateState();
  // clang-format off
  const float matrix_hi[] = {1.0 / std::sqrt(2), 0.0, 0.0, 0.0, 1.0 / std::sqrt(2), 0.0, 0.0, 0.0,
                             0.0, 0.0, 1.0 / std::sqrt(2), 0.0, 0.0, 0.0, 1.0 / std::sqrt(2), 0.0,
                             1.0 / std::sqrt(2), 0.0, 0.0, 0.0, -1.0 / std::sqrt(2), 0.0, 0.0, 0.0,
                             0.0, 0.0, 1.0 / std::sqrt(2), 0.0, 0.0, 0.0,-1.0 / std::sqrt(2), 0.0};
  // clang-format on
  const float matrix_cnot[] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  state->SetStateZero();
  // Apply H_0I_3, taking |000000> to (1/sqrt(2))|000000> + (1/sqrt(2))|100000>
  state->ApplyGate2(0, 3, matrix_hi);
  ASSERT_EQ(state->GetAmpl(0), std::complex<float>(1 / std::sqrt(2), 0.0));
  ASSERT_EQ(state->GetAmpl(32), std::complex<float>(1 / std::sqrt(2), 0.0));
  // Apply CNOT_03, taking (1/sqrt(2))|000000> + (1/sqrt(2))|100000>
  // to (1/sqrt(2))|000000> + (1/sqrt(2))|100100>
  state->ApplyGate2(0, 3, matrix_cnot);
  ASSERT_EQ(state->GetAmpl(0), std::complex<float>(1 / std::sqrt(2), 0.0));
  ASSERT_EQ(state->GetAmpl(36), std::complex<float>(1 / std::sqrt(2), 0.0));
}

TEST(StateSpaceTest, Update) {
  const std::array<float, 8> matrix_h = {
      1.0 / std::sqrt(2), 0.0, 1.0 / std::sqrt(2),  0.0,
      1.0 / std::sqrt(2), 0.0, -1.0 / std::sqrt(2), 0.0};
  Gate gate_small(0, 0, matrix_h);
  std::vector<Gate> gates_small;
  gates_small.push_back(gate_small);
  const Circuit circuit_small(1, gates_small);
  auto state_small = std::unique_ptr<StateSpace>(GetStateSpace(1, 1));
  state_small->CreateState();
  state_small->SetStateZero();
  ASSERT_EQ(state_small->Update(circuit_small), tensorflow::Status::OK());
  EXPECT_NEAR(state_small->GetAmpl(0).real(), 1.0 / std::sqrt(2), 1E-5);
  EXPECT_NEAR(state_small->GetAmpl(0).imag(), 0.0, 1E-5);
  EXPECT_NEAR(state_small->GetAmpl(1).real(), 1.0 / std::sqrt(2), 1E-5);
  EXPECT_NEAR(state_small->GetAmpl(1).imag(), 0.0, 1E-5);

  // TODO(zaqqwerty): Remove identity gates once fuser is updated (#171)
  const uint64_t num_qubits(7);
  const uint64_t q0(2);
  const uint64_t q1(5);
  Gate gate_0(0, q0, matrix_h);
  Gate gate_1(1, q1, matrix_h);
  const std::array<float, 32> matrix_i = {
      1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  Gate gate_i(2, q0, q1, matrix_i);
  std::vector<Gate> gates;
  gates.push_back(gate_0);
  gates.push_back(gate_1);
  gates.push_back(gate_i);
  const Circuit circuit(num_qubits, gates);
  auto state = std::unique_ptr<StateSpace>(GetStateSpace(num_qubits, 1));
  state->CreateState();
  state->SetStateZero();
  ASSERT_EQ(state->Update(circuit), tensorflow::Status::OK());
  for (uint64_t i = 0; i < state_small->GetDimension(); i++) {
    if (i == 0 || i == 1 << q0 || i == 1 << q1 || i == (1 << q0) + (1 << q1)) {
      EXPECT_NEAR(state->GetAmpl(i).real(), 0.5, 1E-5);
      EXPECT_NEAR(state->GetAmpl(i).imag(), 0.0, 1E-5);
    } else {
      EXPECT_NEAR(state->GetAmpl(i).real(), 0.0, 1E-5);
      EXPECT_NEAR(state->GetAmpl(i).imag(), 0.0, 1E-5);
    }
  }
}

TEST(StateSpaceTest, ComputeExpectation) {
  const uint64_t num_qubits(9);
  const uint64_t q0(3);
  const uint64_t q1(7);
  auto state = std::unique_ptr<StateSpace>(GetStateSpace(num_qubits, 1));
  auto scratch = std::unique_ptr<StateSpace>(GetStateSpace(num_qubits, 1));
  state->CreateState();
  scratch->CreateState();

  float zz_coeff(5.95);
  float xx_coeff(-3.4);
  tfq::proto::PauliSum p_sum_zz, p_sum_xx;

  // Initialize ZZ
  tfq::proto::PauliTerm* p_term_scratch_zz = p_sum_zz.add_terms();
  p_term_scratch_zz->set_coefficient_real(zz_coeff);
  tfq::proto::PauliQubitPair* pair_proto_zz = p_term_scratch_zz->add_paulis();
  // NOTE: qubit endianess is opposite that of raw gates
  pair_proto_zz->set_qubit_id(std::to_string(q0));
  pair_proto_zz->set_pauli_type("Z");
  pair_proto_zz = p_term_scratch_zz->add_paulis();
  pair_proto_zz->set_qubit_id(std::to_string(q1));
  pair_proto_zz->set_pauli_type("Z");

  // Initialize XX
  tfq::proto::PauliTerm* p_term_scratch_xx = p_sum_xx.add_terms();
  p_term_scratch_xx->set_coefficient_real(xx_coeff);
  tfq::proto::PauliQubitPair* pair_proto_xx = p_term_scratch_xx->add_paulis();
  pair_proto_xx->set_qubit_id(std::to_string(q0));
  pair_proto_xx->set_pauli_type("X");
  pair_proto_xx = p_term_scratch_xx->add_paulis();
  pair_proto_xx->set_qubit_id(std::to_string(q1));
  pair_proto_xx->set_pauli_type("X");

  // Check expectation values on vacuum
  state->SetStateZero();
  float expectation_value_zz(0);
  float expectation_value_xx(0);
  ASSERT_EQ(
      state->ComputeExpectation(p_sum_zz, scratch.get(), &expectation_value_zz),
      tensorflow::Status::OK());
  ASSERT_EQ(
      state->ComputeExpectation(p_sum_xx, scratch.get(), &expectation_value_xx),
      tensorflow::Status::OK());
  EXPECT_NEAR(expectation_value_zz, zz_coeff, 1E-5);
  EXPECT_NEAR(expectation_value_xx, 0.0, 1E-5);

  // Check expectation values on bit flips
  // |...1...0...>
  state->SetStateZero();
  expectation_value_zz = 0;
  expectation_value_xx = 0;
  state->SetAmpl(0, std::complex<float>(0., 0.));
  state->SetAmpl(1 << (num_qubits - q0 - 1), std::complex<float>(1., 0.));
  ASSERT_EQ(
      state->ComputeExpectation(p_sum_zz, scratch.get(), &expectation_value_zz),
      tensorflow::Status::OK());
  ASSERT_EQ(
      state->ComputeExpectation(p_sum_xx, scratch.get(), &expectation_value_xx),
      tensorflow::Status::OK());
  EXPECT_NEAR(expectation_value_zz, -1.0 * zz_coeff, 1E-5);
  EXPECT_NEAR(expectation_value_xx, 0.0, 1E-5);
  // |...0...1...>
  state->SetStateZero();
  expectation_value_zz = 0;
  expectation_value_xx = 0;
  state->SetAmpl(0, std::complex<float>(0., 0.));
  state->SetAmpl(1 << (num_qubits - q1 - 1), std::complex<float>(1., 0.));
  ASSERT_EQ(
      state->ComputeExpectation(p_sum_zz, scratch.get(), &expectation_value_zz),
      tensorflow::Status::OK());
  ASSERT_EQ(
      state->ComputeExpectation(p_sum_xx, scratch.get(), &expectation_value_xx),
      tensorflow::Status::OK());
  EXPECT_NEAR(expectation_value_zz, -1.0 * zz_coeff, 1E-5);
  EXPECT_NEAR(expectation_value_xx, 0.0, 1E-5);
  // |...1...1...>
  state->SetStateZero();
  expectation_value_zz = 0;
  expectation_value_xx = 0;
  state->SetAmpl(0, std::complex<float>(0., 0.));
  state->SetAmpl((1 << (num_qubits - q0 - 1)) + (1 << (num_qubits - q1 - 1)),
                 std::complex<float>(1., 0.));
  ASSERT_EQ(
      state->ComputeExpectation(p_sum_zz, scratch.get(), &expectation_value_zz),
      tensorflow::Status::OK());
  ASSERT_EQ(
      state->ComputeExpectation(p_sum_xx, scratch.get(), &expectation_value_xx),
      tensorflow::Status::OK());
  EXPECT_NEAR(expectation_value_zz, zz_coeff, 1E-5);
  EXPECT_NEAR(expectation_value_xx, 0.0, 1E-5);

  // Check expectation values on phase flips
  // |...+...+...>
  state->SetStateZero();
  expectation_value_zz = 0;
  expectation_value_xx = 0;
  state->SetAmpl(0, std::complex<float>(0.5, 0.));
  state->SetAmpl(1 << (num_qubits - q0 - 1), std::complex<float>(0.5, 0.));
  state->SetAmpl(1 << (num_qubits - q1 - 1), std::complex<float>(0.5, 0.));
  state->SetAmpl((1 << (num_qubits - q0 - 1)) + (1 << (num_qubits - q1 - 1)),
                 std::complex<float>(0.5, 0.));
  ASSERT_EQ(
      state->ComputeExpectation(p_sum_zz, scratch.get(), &expectation_value_zz),
      tensorflow::Status::OK());
  ASSERT_EQ(
      state->ComputeExpectation(p_sum_xx, scratch.get(), &expectation_value_xx),
      tensorflow::Status::OK());
  EXPECT_NEAR(expectation_value_zz, 0.0, 1E-5);
  EXPECT_NEAR(expectation_value_xx, xx_coeff, 1E-5);
  // |...-...+...>
  state->SetStateZero();
  expectation_value_zz = 0;
  expectation_value_xx = 0;
  state->SetAmpl(0, std::complex<float>(0.5, 0.));
  state->SetAmpl(1 << (num_qubits - q0 - 1), std::complex<float>(-0.5, 0.));
  state->SetAmpl(1 << (num_qubits - q1 - 1), std::complex<float>(0.5, 0.));
  state->SetAmpl((1 << (num_qubits - q0 - 1)) + (1 << (num_qubits - q1 - 1)),
                 std::complex<float>(-0.5, 0.));
  ASSERT_EQ(
      state->ComputeExpectation(p_sum_zz, scratch.get(), &expectation_value_zz),
      tensorflow::Status::OK());
  ASSERT_EQ(
      state->ComputeExpectation(p_sum_xx, scratch.get(), &expectation_value_xx),
      tensorflow::Status::OK());
  EXPECT_NEAR(expectation_value_zz, 0.0, 1E-5);
  EXPECT_NEAR(expectation_value_xx, -1.0 * xx_coeff, 1E-5);
  // |...+...-...>
  state->SetStateZero();
  expectation_value_zz = 0;
  expectation_value_xx = 0;
  state->SetAmpl(0, std::complex<float>(0.5, 0.));
  state->SetAmpl(1 << (num_qubits - q0 - 1), std::complex<float>(0.5, 0.));
  state->SetAmpl(1 << (num_qubits - q1 - 1), std::complex<float>(-0.5, 0.));
  state->SetAmpl((1 << (num_qubits - q0 - 1)) + (1 << (num_qubits - q1 - 1)),
                 std::complex<float>(-0.5, 0.));
  ASSERT_EQ(
      state->ComputeExpectation(p_sum_zz, scratch.get(), &expectation_value_zz),
      tensorflow::Status::OK());
  ASSERT_EQ(
      state->ComputeExpectation(p_sum_xx, scratch.get(), &expectation_value_xx),
      tensorflow::Status::OK());
  EXPECT_NEAR(expectation_value_zz, 0.0, 1E-5);
  EXPECT_NEAR(expectation_value_xx, -1.0 * xx_coeff, 1E-5);
  // |...-...-...>
  state->SetStateZero();
  expectation_value_zz = 0;
  expectation_value_xx = 0;
  state->SetAmpl(0, std::complex<float>(0.5, 0.));
  state->SetAmpl(1 << (num_qubits - q0 - 1), std::complex<float>(-0.5, 0.));
  state->SetAmpl(1 << (num_qubits - q1 - 1), std::complex<float>(-0.5, 0.));
  state->SetAmpl((1 << (num_qubits - q0 - 1)) + (1 << (num_qubits - q1 - 1)),
                 std::complex<float>(0.5, 0.));
  ASSERT_EQ(
      state->ComputeExpectation(p_sum_zz, scratch.get(), &expectation_value_zz),
      tensorflow::Status::OK());
  ASSERT_EQ(
      state->ComputeExpectation(p_sum_xx, scratch.get(), &expectation_value_xx),
      tensorflow::Status::OK());
  EXPECT_NEAR(expectation_value_zz, 0.0, 1E-5);
  EXPECT_NEAR(expectation_value_xx, xx_coeff, 1E-5);
}

TEST(StateSpaceTest, SampleStateOneSample) {
  auto equal = std::unique_ptr<StateSpace>(GetStateSpace(1, 1));
  equal->CreateState();
  equal->SetAmpl(0, std::complex<float>(1.0, 0.));
  equal->SetAmpl(1, std::complex<float>(0.0, 0.));

  std::vector<uint64_t> samples;
  equal->SampleState(1, &samples);
  ASSERT_EQ(samples.size(), 1);
}

TEST(StateSpaceTest, SampleStateZeroSamples) {
  auto equal = std::unique_ptr<StateSpace>(GetStateSpace(1, 1));
  equal->CreateState();
  equal->SetAmpl(0, std::complex<float>(1.0, 0.));
  equal->SetAmpl(1, std::complex<float>(0.0, 0.));

  std::vector<uint64_t> samples;
  equal->SampleState(0, &samples);
  ASSERT_EQ(samples.size(), 0);
}

TEST(StateSpaceTest, SampleStateEqual) {
  auto equal = std::unique_ptr<StateSpace>(GetStateSpace(1, 1));
  equal->CreateState();
  equal->SetAmpl(0, std::complex<float>(0.707, 0.));
  equal->SetAmpl(1, std::complex<float>(0.707, 0.));

  std::vector<uint64_t> samples;
  const int m = 100000;
  equal->SampleState(m, &samples);

  float num_ones = 0.0;
  for (int i = 0; i < m; i++) {
    if (samples[i]) {
      num_ones++;
    }
  }
  ASSERT_EQ(samples.size(), m);
  EXPECT_NEAR(num_ones / static_cast<float>(m), 0.5, 1E-2);
}

TEST(StateSpaceTest, SampleStateSkew) {
  auto skew = std::unique_ptr<StateSpace>(GetStateSpace(1, 1));
  skew->CreateState();

  std::vector<float> rots = {0.1, 0.3, 0.5, 0.7, 0.9};
  for (int t = 0; t < 5; t++) {
    float z_amp = std::sqrt(rots[t]);
    float o_amp = std::sqrt(1.0 - rots[t]);
    skew->SetAmpl(0, std::complex<float>(z_amp, 0.));
    skew->SetAmpl(1, std::complex<float>(o_amp, 0.));

    std::vector<uint64_t> samples;
    const int m = 100000;
    skew->SampleState(m, &samples);
    float num_z = 0.0;
    for (int i = 0; i < m; i++) {
      if (samples[i] == 0) {
        num_z++;
      }
    }
    ASSERT_EQ(samples.size(), m);
    EXPECT_NEAR(num_z / static_cast<float>(m), rots[t], 1E-2);
  }
}

TEST(StateSpaceTest, SampleStateComplexDist) {
  auto state = std::unique_ptr<StateSpace>(GetStateSpace(3, 1));
  state->CreateState();

  std::vector<float> probs = {0.05, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2};
  for (int i = 0; i < 8; i++) {
    state->SetAmpl(i, std::complex<float>(std::sqrt(probs[i]), 0.0));
  }

  std::vector<uint64_t> samples;
  const int m = 100000;
  state->SampleState(m, &samples);

  std::vector<float> measure_probs = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int i = 0; i < m; i++) {
    measure_probs[samples[i]] += 1.0;
  }
  for (int i = 0; i < 8; i++) {
    EXPECT_NEAR(measure_probs[i] / static_cast<float>(m), probs[i], 1E-2);
  }
  ASSERT_EQ(samples.size(), m);
}

}  // namespace
}  // namespace qsim
}  // namespace tfq
