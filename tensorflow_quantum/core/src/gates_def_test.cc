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

#include "tensorflow_quantum/core/src/gates_def.h"

#define _USE_MATH_DEFINES
#include <array>
#include <complex>
#include <cstdlib>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status.h"

namespace tfq {
namespace {

TEST(GatesDefTest, SwapQubits) {
  // Conjugation by swap gate:
  //  | 0  1  2  3  |      | 0  2  1  3  |
  //  | 4  5  6  7  |      | 8  10 9  11 |
  //  | 8  9  10 11 | ---> | 4  6  5  7  |
  //  | 12 13 14 15 |      | 12 14 13 15 |
  // clang-format off
  std::array<float, 32> matrix{
    0,  0.5, 1, 1.5, 2, 2.5, 3, 3.5,
    4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5,
    8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5,
    12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5};
  const std::array<float, 32> matrix_swapped{
    0,  0.5, 2, 2.5, 1, 1.5, 3, 3.5,
    8, 8.5, 10, 10.5, 9, 9.5, 11, 11.5,
    4, 4.5, 6, 6.5, 5, 5.5, 7, 7.5,
    12, 12.5, 14, 14.5, 13, 13.5, 15, 15.5};
  // clang-format on
  SwapQubits(matrix);
  for (int i = 0; i < 32; i++) {
    EXPECT_EQ(matrix[i], matrix_swapped[i]);
  }
}

TEST(GatesDefTest, GateConstructors) {
  // Empty gate constructor
  Gate gate0q;
  EXPECT_EQ(gate0q.time, 0);
  EXPECT_EQ(gate0q.num_qubits, 0);

  // One-qubit gate constructor
  const unsigned int time1q = 256;
  const unsigned int qubits1q = 53;
  const std::array<float, 8> matrix1q{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
  Gate gate1q(time1q, qubits1q, matrix1q);
  EXPECT_EQ(gate1q.time, time1q);
  EXPECT_EQ(gate1q.num_qubits, 1);
  EXPECT_EQ(gate1q.qubits[0], qubits1q);
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(gate1q.matrix[i], matrix1q[i]);
  }

  // Two-qubit gate constructor
  const unsigned int time2q = 512;
  const unsigned int qubits2q1 = 53;
  const unsigned int qubits2q2 = 256;
  const std::array<float, 32> matrix2q{
      0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  0.10,
      0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21,
      0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31};
  Gate gate2q(time2q, qubits2q1, qubits2q2, matrix2q);
  EXPECT_EQ(gate2q.time, time2q);
  EXPECT_EQ(gate2q.num_qubits, 2);
  EXPECT_EQ(gate2q.qubits[0], qubits2q1);
  EXPECT_EQ(gate2q.qubits[1], qubits2q2);
  for (int i = 0; i < 32; i++) {
    EXPECT_EQ(gate2q.matrix[i], matrix2q[i]);
  }

  // Confirm swapping in constructor
  // clang-format off
  std::array<float, 32> matrix_original{
    0,  0.5, 1, 1.5, 2, 2.5, 3, 3.5,
    4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5,
    8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5,
    12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5};
  const std::array<float, 32> matrix_swapped{
    0,  0.5, 2, 2.5, 1, 1.5, 3, 3.5,
    8, 8.5, 10, 10.5, 9, 9.5, 11, 11.5,
    4, 4.5, 6, 6.5, 5, 5.5, 7, 7.5,
    12, 12.5, 14, 14.5, 13, 13.5, 15, 15.5};
  // clang-format on
  Gate gate_original(0, 1, 2, matrix_original);
  Gate gate_swapped(0, 2, 1, matrix_swapped);
  EXPECT_EQ(gate_original, gate_swapped);
}

TEST(GatesDefTest, GateEquality) {
  // Empty gate
  Gate test_gate_0q, real_gate_0q;

  test_gate_0q.time = real_gate_0q.time + 1;
  EXPECT_NE(test_gate_0q, real_gate_0q);
  test_gate_0q.time = real_gate_0q.time;

  test_gate_0q.num_qubits = real_gate_0q.num_qubits + 1;
  EXPECT_NE(test_gate_0q, real_gate_0q);
  test_gate_0q.num_qubits = real_gate_0q.num_qubits;

  EXPECT_EQ(test_gate_0q, real_gate_0q);

  // One-qubit gate
  const unsigned int time1q = 1256;
  const unsigned int qubits1q = 153;
  const std::array<float, 8> matrix1q{0.10, 0.11, 0.12, 0.13,
                                      0.14, 0.15, 0.16, 0.17};
  Gate test_gate_1q(time1q, qubits1q, matrix1q);
  Gate real_gate_1q(time1q, qubits1q, matrix1q);

  test_gate_1q.time = real_gate_1q.time + 1;
  ASSERT_NE(test_gate_1q, real_gate_1q);
  test_gate_1q.time = real_gate_1q.time;

  test_gate_1q.num_qubits = real_gate_1q.num_qubits + 1;
  ASSERT_NE(test_gate_1q, real_gate_1q);
  test_gate_1q.num_qubits = real_gate_1q.num_qubits;

  test_gate_1q.qubits[0] = real_gate_1q.qubits[0] + 1;
  ASSERT_NE(test_gate_1q, real_gate_1q);
  test_gate_1q.qubits[0] = real_gate_1q.qubits[0];

  test_gate_1q.matrix[0] = real_gate_1q.matrix[0] + 1;
  ASSERT_NE(test_gate_1q, real_gate_1q);
  test_gate_1q.matrix[0] = real_gate_1q.matrix[0];

  test_gate_1q.matrix[7] = real_gate_1q.matrix[7] + 1;
  ASSERT_NE(test_gate_1q, real_gate_1q);
  test_gate_1q.matrix[7] = real_gate_1q.matrix[7];

  ASSERT_EQ(test_gate_1q, real_gate_1q);

  // Two-qubit gate
  const unsigned int time2q = 2512;
  const unsigned int qubits2q1 = 253;
  const unsigned int qubits2q2 = 2256;
  const std::array<float, 32> matrix2q{
      0.20,  0.21,  0.22,  0.23,  0.24,  0.25,  0.26,  0.27,
      0.28,  0.29,  0.210, 0.211, 0.212, 0.213, 0.214, 0.215,
      0.216, 0.217, 0.218, 0.219, 0.220, 0.221, 0.223, 0.224,
      0.225, 0.226, 0.227, 0.228, 0.229, 0.230, 0.231};
  Gate test_gate_2q(time2q, qubits2q1, qubits2q2, matrix2q);
  Gate real_gate_2q(time2q, qubits2q1, qubits2q2, matrix2q);

  test_gate_2q.time = real_gate_2q.time + 1;
  ASSERT_NE(test_gate_2q, real_gate_2q);
  test_gate_2q.time = real_gate_2q.time;

  test_gate_2q.num_qubits = real_gate_2q.num_qubits + 1;
  ASSERT_NE(test_gate_2q, real_gate_2q);
  test_gate_2q.num_qubits = real_gate_2q.num_qubits;

  test_gate_2q.qubits[0] = real_gate_2q.qubits[0] + 1;
  ASSERT_NE(test_gate_2q, real_gate_2q);
  test_gate_2q.qubits[0] = real_gate_2q.qubits[0];

  test_gate_2q.qubits[1] = real_gate_2q.qubits[1] + 1;
  ASSERT_NE(test_gate_2q, real_gate_2q);
  test_gate_2q.qubits[1] = real_gate_2q.qubits[1];

  test_gate_2q.matrix[0] = real_gate_2q.matrix[0] + 1;
  ASSERT_NE(test_gate_2q, real_gate_2q);
  test_gate_2q.matrix[0] = real_gate_2q.matrix[0];

  test_gate_2q.matrix[31] = real_gate_2q.matrix[31] + 1;
  ASSERT_NE(test_gate_2q, real_gate_2q);
  test_gate_2q.matrix[31] = real_gate_2q.matrix[31];

  ASSERT_EQ(test_gate_2q, real_gate_2q);
}

// ============================================================================
// GateBuilder interface tests.
// ============================================================================

TEST(GatesDefTest, GateBuilder) {
  const unsigned int time_1q = 15;
  const unsigned int qubit_1q = 53;
  const std::array<float, 8> matrix_1q{0, 1, 2, 3, 4, 5, 6, 7};

  class ConstantGateBuilder : public GateBuilder {
   public:
    virtual tensorflow::Status Build(
        const unsigned int time, const std::vector<unsigned int>& locations,
        const absl::flat_hash_map<std::string, float>& args,
        Gate* gate) override {
      const std::array<float, 8> matrix_1q_internal{0, 1, 2, 3, 4, 5, 6, 7};
      *gate = Gate(time_1q, qubit_1q, matrix_1q_internal);
      return tensorflow::Status::OK();
    }
  };

  ConstantGateBuilder test_builder;
  Gate test_gate;
  const unsigned int time_ignored = 4444;
  ASSERT_EQ(
      test_builder.Build(time_ignored, std::vector<unsigned int>(),
                         absl::flat_hash_map<std::string, float>(), &test_gate),
      tensorflow::Status::OK());
  ASSERT_EQ(test_gate, Gate(time_1q, qubit_1q, matrix_1q));
}

// ============================================================================
// GateBuilder implementation tests.
// ============================================================================

TEST(GatesDefTest, XPow) {
  XPowGateBuilder builder;
  const unsigned int time{3};
  const unsigned int qubit{53};
  std::vector<unsigned int> locations;
  locations.push_back(qubit);

  // cirq X gate is XPowGate at exponent of 1.
  std::array<float, 8> matrix{0, 0, 1, 0, 1, 0, 0, 0};
  Gate real_gate(time, qubit, matrix);
  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = 0.0;
  arg_map["exponent"] = 1.0;
  arg_map["exponent_scalar"] = 1.0;
  Gate test_gate;
  ASSERT_EQ(builder.Build(time, locations, arg_map, &test_gate),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate, real_gate);

  // RX gates are XPow gates with global shift of -0.5
  for (auto const& angle : {0.1234, 5.4321}) {
    std::array<float, 8> matrix_rot{
        (float)std::cos(angle / 2.),  0, 0,
        (float)-std::sin(angle / 2.), 0, (float)-std::sin(angle / 2.),
        (float)std::cos(angle / 2.),  0};
    Gate real_gate_rot(time, qubit, matrix_rot);
    absl::flat_hash_map<std::string, float> arg_map_rot;
    arg_map_rot["global_shift"] = -0.5;
    arg_map_rot["exponent"] = angle / M_PI;
    arg_map_rot["exponent_scalar"] = 1.0;
    Gate test_gate_rot;
    ASSERT_EQ(builder.Build(time, locations, arg_map_rot, &test_gate_rot),
              tensorflow::Status::OK());
    ASSERT_EQ(test_gate_rot, real_gate_rot);
  }
}

TEST(GatesDefTest, YPow) {
  YPowGateBuilder builder;
  const unsigned int time{3};
  const unsigned int qubit{53};
  std::vector<unsigned int> locations;
  locations.push_back(qubit);

  // cirq Y gate is YPowGate at exponent of 1.
  std::array<float, 8> matrix{0, 0, 0, -1, 0, 1, 0, 0};
  Gate real_gate(time, qubit, matrix);
  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = 0.0;
  arg_map["exponent"] = 1.0;
  arg_map["exponent_scalar"] = 1.0;
  Gate test_gate;
  ASSERT_EQ(builder.Build(time, locations, arg_map, &test_gate),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate, real_gate);

  // RY gates are YPow gates with global shift of -0.5
  for (auto const& angle : {0.1234, 5.4321}) {
    std::array<float, 8> matrix_rot{
        (float)std::cos(angle / 2.), 0, (float)-std::sin(angle / 2.), 0,
        (float)std::sin(angle / 2.), 0, (float)std::cos(angle / 2.),  0};
    Gate real_gate_rot(time, qubit, matrix_rot);
    absl::flat_hash_map<std::string, float> arg_map_rot;
    arg_map_rot["global_shift"] = -0.5;
    arg_map_rot["exponent"] = angle / M_PI;
    arg_map_rot["exponent_scalar"] = 1.0;
    Gate test_gate_rot;
    ASSERT_EQ(builder.Build(time, locations, arg_map_rot, &test_gate_rot),
              tensorflow::Status::OK());
    ASSERT_EQ(test_gate_rot, real_gate_rot);
  }
}

TEST(GatesDefTest, ZPow) {
  ZPowGateBuilder builder;
  const unsigned int time{3};
  const unsigned int qubit{53};
  std::vector<unsigned int> locations;
  locations.push_back(qubit);

  // cirq Z gate is ZPowGate at exponent of 1.
  std::array<float, 8> matrix{1, 0, 0, 0, 0, 0, -1, 0};
  Gate real_gate(time, qubit, matrix);
  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = 0.0;
  arg_map["exponent"] = 1.0;
  arg_map["exponent_scalar"] = 1.0;
  Gate test_gate;
  ASSERT_EQ(builder.Build(time, locations, arg_map, &test_gate),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate, real_gate);

  // S gate is ZPowGate with exponent of 0.5
  std::array<float, 8> matrix_s{1.0, 0, 0, 0, 0, 0, 0, 1.0};
  Gate real_gate_s(time, qubit, matrix_s);
  absl::flat_hash_map<std::string, float> arg_map_s;
  arg_map_s["global_shift"] = 0.0;
  arg_map_s["exponent"] = 0.5;
  arg_map_s["exponent_scalar"] = 1.0;
  Gate test_gate_s;
  ASSERT_EQ(builder.Build(time, locations, arg_map_s, &test_gate_s),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate_s, real_gate_s);

  // T gate is ZPowGate with exponent of 0.25
  std::array<float, 8> matrix_tg{
      1.0, 0, 0, 0, 0, 0, 1 / std::sqrt(2), 1 / std::sqrt(2)};
  Gate real_gate_tg(time, qubit, matrix_tg);
  absl::flat_hash_map<std::string, float> arg_map_tg;
  arg_map_tg["global_shift"] = 0.0;
  arg_map_tg["exponent"] = 0.25;
  arg_map_tg["exponent_scalar"] = 1.0;
  Gate test_gate_tg;
  ASSERT_EQ(builder.Build(time, locations, arg_map_tg, &test_gate_tg),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate_tg, real_gate_tg);

  // RZ gates are ZPow gates with global shift of -0.5
  for (auto const& angle : {0.1234, 5.4321}) {
    std::complex<double> m_00;
    m_00 = std::exp(std::complex<double>(0, -1.0 * angle / 2.));
    std::complex<double> m_11;
    m_11 = std::exp(std::complex<double>(0, angle / 2.));
    std::array<float, 8> matrix_rot{
        (float)m_00.real(), (float)m_00.imag(), 0, 0, 0, 0,
        (float)m_11.real(), (float)m_11.imag()};
    Gate real_gate_rot(time, qubit, matrix_rot);
    absl::flat_hash_map<std::string, float> arg_map_rot;
    arg_map_rot["global_shift"] = -0.5;
    arg_map_rot["exponent"] = angle / M_PI;
    arg_map_rot["exponent_scalar"] = 1.0;
    Gate test_gate_rot;
    ASSERT_EQ(builder.Build(time, locations, arg_map_rot, &test_gate_rot),
              tensorflow::Status::OK());
    ASSERT_EQ(test_gate_rot, real_gate_rot);
  }
}

TEST(GatesDefTest, HPow) {
  HPowGateBuilder builder;
  const unsigned int time{3};
  const unsigned int qubit{53};
  std::vector<unsigned int> locations;
  locations.push_back(qubit);

  // cirq H gate is HPowGate at exponent of 1.
  std::array<float, 8> matrix{1 / std::sqrt(2), 0, 1 / std::sqrt(2),  0,
                              1 / std::sqrt(2), 0, -1 / std::sqrt(2), 0};
  Gate real_gate(time, qubit, matrix);
  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = 0.0;
  arg_map["exponent"] = 1.0;
  arg_map["exponent_scalar"] = 1.0;
  Gate test_gate;
  ASSERT_EQ(builder.Build(time, locations, arg_map, &test_gate),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate, real_gate);
}

TEST(GatesDefTest, IdentityGate) {
  IGateBuilder builder;
  const unsigned int time{3};
  const unsigned int qubit{53};
  std::vector<unsigned int> locations;
  locations.push_back(qubit);

  std::array<float, 8> matrix{1, 0, 0, 0, 0, 0, 1, 0};
  Gate real_gate(time, qubit, matrix);
  absl::flat_hash_map<std::string, float> arg_map;
  Gate test_gate;
  ASSERT_EQ(builder.Build(time, locations, arg_map, &test_gate),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate, real_gate);
}

TEST(GatesDefTest, PhasedXPow) {
  PhasedXPowGateBuilder builder;
  const unsigned int time{3};
  const unsigned int qubit{53};
  std::vector<unsigned int> locations;
  locations.push_back(qubit);

  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = -0.2;
  arg_map["exponent"] = 1.7;
  arg_map["exponent_scalar"] = 1.0;
  arg_map["phase_exponent"] = 1.1;
  arg_map["phase_exponent_scalar"] = 1.0;
  Gate test_gate;
  ASSERT_EQ(builder.Build(time, locations, arg_map, &test_gate),
            tensorflow::Status::OK());

  // Associated matrix elements for above parameters extracted using cirq
  std::array<float, 8> matrix{0.02798719, -0.89056687, -0.43596421,
                              0.12665931, -0.42715093, -0.1537838,
                              0.02798719, -0.89056687};
  Gate real_gate(time, qubit, matrix);

  ASSERT_EQ(test_gate, real_gate);
}

TEST(GatesDefTest, XXPow) {
  XXPowGateBuilder builder;
  const unsigned int time{3};
  const unsigned int q1{53};
  const unsigned int q2{55};
  std::vector<unsigned int> locations;
  locations.push_back(q1);
  locations.push_back(q2);

  // cirq XX gate is XXPowGate at exponent of 1
  // clang-format off
  std::array<float, 32> matrix{0, 0, 0, 0, 0, 0, 1, 0,
                               0, 0, 0, 0, 1, 0, 0, 0,
                               0, 0, 1, 0, 0, 0, 0, 0,
                               1, 0, 0, 0, 0, 0, 0, 0};
  // clang-format on
  Gate real_gate(time, q1, q2, matrix);
  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = 0.0;
  arg_map["exponent"] = 1.0;
  arg_map["exponent_scalar"] = 1.0;
  Gate test_gate;
  ASSERT_EQ(builder.Build(time, locations, arg_map, &test_gate),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate, real_gate);
}

TEST(GatesDefTest, YYPow) {
  YYPowGateBuilder builder;
  const unsigned int time{3};
  const unsigned int q1{53};
  const unsigned int q2{55};
  std::vector<unsigned int> locations;
  locations.push_back(q1);
  locations.push_back(q2);

  // cirq YY gate is YYPowGate at exponent of 1
  // clang-format off
  std::array<float, 32> matrix{0, 0, 0, 0, 0, 0, -1, 0,
                               0, 0, 0, 0, 1, 0, 0, 0,
                               0, 0, 1, 0, 0, 0, 0, 0,
                               -1, 0, 0, 0, 0, 0, 0, 0};
  // clang-format on
  Gate real_gate(time, q1, q2, matrix);
  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = 0.0;
  arg_map["exponent"] = 1.0;
  arg_map["exponent_scalar"] = 1.0;
  Gate test_gate;
  ASSERT_EQ(builder.Build(time, locations, arg_map, &test_gate),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate, real_gate);
}

TEST(GatesTest, ZZPow) {
  ZZPowGateBuilder builder;
  const unsigned int time{3};
  const unsigned int q1{53};
  const unsigned int q2{55};
  std::vector<unsigned int> locations;
  locations.push_back(q1);
  locations.push_back(q2);

  // cirq ZZ gate is ZZPowGate at exponent of 1
  // clang-format off
  std::array<float, 32> matrix{1, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, -1, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, -1, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 1, 0};
  // clang-format on
  Gate real_gate(time, q1, q2, matrix);
  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = 0.0;
  arg_map["exponent"] = 1.0;
  arg_map["exponent_scalar"] = 1.0;
  Gate test_gate;
  ASSERT_EQ(builder.Build(time, locations, arg_map, &test_gate),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate, real_gate);
}

TEST(GatesDefTest, CZPow) {
  CZPowGateBuilder builder;
  const unsigned int time{3};
  const unsigned int q1{53};
  const unsigned int q2{55};
  std::vector<unsigned int> locations;
  locations.push_back(q1);
  locations.push_back(q2);

  // CZ gate is CZPowGate at exponent of 1
  // clang-format off
  std::array<float, 32> matrix{1, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 1, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 1, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, -1, 0};
  // clang-format on
  Gate real_gate(time, q1, q2, matrix);
  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = 0.0;
  arg_map["exponent"] = 1.0;
  arg_map["exponent_scalar"] = 1.0;
  Gate test_gate;
  ASSERT_EQ(builder.Build(time, locations, arg_map, &test_gate),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate, real_gate);
}

TEST(GatesDefTest, CNotPow) {
  CNotPowGateBuilder builder;
  const unsigned int time{3};
  const unsigned int q1{53};
  const unsigned int q2{55};
  std::vector<unsigned int> locations;
  locations.push_back(q1);
  locations.push_back(q2);

  // CNot gate is CNotPowGate at exponent of 1
  // clang-format off
  std::array<float, 32> matrix{1, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 1, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 1, 0,
                               0, 0, 0, 0, 1, 0, 0, 0};
  // clang-format on
  Gate real_gate(time, q1, q2, matrix);
  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = 0.0;
  arg_map["exponent"] = 1.0;
  arg_map["exponent_scalar"] = 1.0;
  Gate test_gate;
  ASSERT_EQ(builder.Build(time, locations, arg_map, &test_gate),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate, real_gate);
}

TEST(GatesDefTest, SwapPow) {
  SwapPowGateBuilder builder;
  const unsigned int time{3};
  const unsigned int q1{53};
  const unsigned int q2{55};
  std::vector<unsigned int> locations;
  locations.push_back(q1);
  locations.push_back(q2);

  // Swap gate is SwapPowGate at exponent of 1
  // clang-format off
  std::array<float, 32> matrix{1, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 1, 0, 0, 0,
                               0, 0, 1, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 1, 0};
  // clang-format on
  Gate real_gate(time, q1, q2, matrix);
  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = 0.0;
  arg_map["exponent"] = 1.0;
  arg_map["exponent_scalar"] = 1.0;
  Gate test_gate;
  ASSERT_EQ(builder.Build(time, locations, arg_map, &test_gate),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate, real_gate);
}

TEST(GatesDefTest, ISwapPow) {
  ISwapPowGateBuilder builder;
  const unsigned int time{3};
  const unsigned int q1{53};
  const unsigned int q2{55};
  std::vector<unsigned int> locations;
  locations.push_back(q1);
  locations.push_back(q2);

  // ISwap gate is ISwapPowGate at exponent of 1
  // clang-format off
  std::array<float, 32> matrix{1, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 1, 0, 0,
                               0, 0, 0, 1, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 1, 0};
  // clang-format on
  Gate real_gate(time, q1, q2, matrix);
  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = 0.0;
  arg_map["exponent"] = 1.0;
  arg_map["exponent_scalar"] = 1.0;
  Gate test_gate;
  ASSERT_EQ(builder.Build(time, locations, arg_map, &test_gate),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate, real_gate);
}

TEST(GatesDefTest, PhasedISwapPow) {
  PhasedISwapPowGateBuilder builder;
  const unsigned int time{3};
  const unsigned int q1{53};
  const unsigned int q2{55};
  std::vector<unsigned int> locations;
  locations.push_back(q1);
  locations.push_back(q2);

  // ISwap gate is PhasedISwapPowGate at exponent of 1
  // and phase_exponent of 0.
  // clang-format off
  std::array<float, 32> matrix{1, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 1, 0, 0,
                               0, 0, 0, 1, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 1, 0};
  // clang-format on
  Gate real_gate(time, q1, q2, matrix);
  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = 0.0;
  arg_map["exponent"] = 1.0;
  arg_map["exponent_scalar"] = 1.0;
  arg_map["phase_exponent"] = 0.0;
  arg_map["phase_exponent_scalar"] = 1.0;
  Gate test_gate;
  ASSERT_EQ(builder.Build(time, locations, arg_map, &test_gate),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate, real_gate);
}

TEST(GatesDefTest, I2) {
  I2GateBuilder builder;
  const unsigned int time{3};
  const unsigned int q1{53};
  const unsigned int q2{55};
  std::vector<unsigned int> locations;
  locations.push_back(q1);
  locations.push_back(q2);

  // clang-format off
  std::array<float, 32> matrix{1, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 1, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 1, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 1, 0};
  // clang-format on
  Gate real_gate(time, q1, q2, matrix);
  absl::flat_hash_map<std::string, float> arg_map;
  Gate test_gate;
  ASSERT_EQ(builder.Build(time, locations, arg_map, &test_gate),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate, real_gate);
}

TEST(GatesDefTest, FSim) {
  FSimGateBuilder builder;
  const unsigned int time{3};
  const unsigned int q1{53};
  const unsigned int q2{55};
  std::vector<unsigned int> locations;
  locations.push_back(q1);
  locations.push_back(q2);

  // FSimGate has limiting forms of iSWAP and CZ, with some relative phasing.
  const std::array<float, 2> angle_pair_1{M_PI / 2, 0};
  const std::array<float, 2> angle_pair_2{0, M_PI};
  const std::array<float, 2> angle_pair_3{M_PI / 2, M_PI / 6};
  const std::array<std::array<float, 2>, 3> angles{angle_pair_1, angle_pair_2,
                                                   angle_pair_3};

  // clang-format off
  const std::array<float, 32> matrix_1{1, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, -1, 0, 0,
                                       0, 0, 0, -1, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 1, 0};
  const std::array<float, 32> matrix_2{1, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 1, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 1, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, -1, 0};
  const std::array<float, 32> matrix_3{1, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, -1, 0, 0,
                                       0, 0, 0, -1, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, std::sqrt(3)/2, -1.0/2};
  // clang-format on
  const std::array<std::array<float, 32>, 3> matrices{matrix_1, matrix_2,
                                                      matrix_3};

  for (long unsigned int i = 0; i < angles.size(); i++) {
    Gate real_gate(time, q1, q2, matrices.at(i));
    absl::flat_hash_map<std::string, float> arg_map;
    arg_map["theta"] = angles.at(i).at(0);
    arg_map["theta_scalar"] = 1.0;
    arg_map["phi"] = angles.at(i).at(1);
    arg_map["phi_scalar"] = 1.0;
    Gate test_gate;
    ASSERT_EQ(builder.Build(time, locations, arg_map, &test_gate),
              tensorflow::Status::OK());
    ASSERT_EQ(test_gate, real_gate);
  }
}

}  // namespace
}  // namespace tfq
