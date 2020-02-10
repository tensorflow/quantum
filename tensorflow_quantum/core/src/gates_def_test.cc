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

#include <array>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status.h"

namespace tfq {
namespace {

TEST(GatesDefTest, GateBuilder) {
  const unsigned int time_1q = 15;
  const unsigned int qubit_1q = 53;
  const std::array<float, 8> matrix_1q{0, 1, 2, 3, 4, 5, 6, 7};
  class ConstantGateBuilder : public GateBuilder {
    virtual tensorflow::Status Build(
        const unsigned int time, const std::vector<unsigned int>& locations,
        const absl::flat_hash_map<std::string, float>& args, Gate* gate) override {
      *gate = Gate(time_1q, qubit_1q, matrix_1q);
      return tensorflow::Status::OK();
    }
  };

  ConstantGateBuilder test_builder;
  Gate test_gate;
  ASSERT_EQ(test_builder.Build(
      unsigned int, std::vector<unsigned int>,
      absl::flast_hash_map<std::string, float>, &test_gate),
            tensorflow::Status::OK());
  ASSERT_EQ(test_gate, Gate(time_1q, qubit_1q, matrix_1q));
}

const double ABS_TOL = 1.0e-14;

void gate_test_func_2cd(Eigen::Matrix2cd gate, Eigen::Matrix2cd gate_test) {
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      EXPECT_NEAR(gate(i, j).real(), gate_test(i, j).real(), ABS_TOL);
      EXPECT_NEAR(gate(i, j).imag(), gate_test(i, j).imag(), ABS_TOL);
    }
  }
}

void gate_test_func_4cd(Eigen::Matrix4cd gate, Eigen::Matrix4cd gate_test) {
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      EXPECT_NEAR(gate(i, j).real(), gate_test(i, j).real(), ABS_TOL);
      EXPECT_NEAR(gate(i, j).imag(), gate_test(i, j).imag(), ABS_TOL);
    }
  }
}

// ============================================================================
// GateBuilder implementation tests.
// ============================================================================

TEST(GatesDefTest, IdentityGate){
  Eigen::Matrix2cd gate = IdentityGate().GetMatrix();
  Eigen::Matrix2cd gate_test;
  gate_test << 1, 0,
    0, 1;
  gate_test_func_2cd(gate, gate_test);
}

TEST(GatesDefTest, CNotGate){
  Eigen::Matrix4cd gate = CNotGate().GetMatrix();
  Eigen::Matrix4cd gate_test;
  gate_test << 1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
    0, 0, 1, 0;
  gate_test_func_4cd(gate, gate_test);
}

// cirq X gate is XPowGate at exponent of 1.
TEST(GatesDefTest, X){
  const auto gate = XPowGate().GetMatrix(1, 0);
  Eigen::Matrix2cd gate_test;
  gate_test << 0, 1, 1, 0;
  gate_test_func_2cd(gate, gate_test);
}

// cirq Y gate is YPowGate at exponent of 1.
TEST(GatesDefTest, Y){
  const auto gate = YPowGate().GetMatrix(1, 0);
  Eigen::Matrix2cd gate_test;
  gate_test << 0, std::complex<double>(0, -1), std::complex<double>(0, 1), 0;
  gate_test_func_2cd(gate, gate_test);
}

// cirq Z gate is ZPowGate at exponent of 1.
TEST(GatesDefTest, Z){
  const auto gate = ZPowGate().GetMatrix(1, 0);
  Eigen::Matrix2cd gate_test;
  gate_test << 1, 0, 0, -1;
  gate_test_func_2cd(gate, gate_test);
}

// cirq H gate is HPowGate at exponent of 1.
TEST(GatesDefTest, H){
  const auto gate = HPowGate().GetMatrix(1, 0);
  Eigen::Matrix2cd gate_test;
  gate_test << 1 / std::sqrt(2), 1 / std::sqrt(2), 1 / std::sqrt(2),
      -1 / std::sqrt(2);
  gate_test_func_2cd(gate, gate_test);
}

// S gate is ZPowGate with exponent of 0.5
TEST(GatesDefTest, S){
  const auto gate = ZPowGate().GetMatrix(0.5, 0);
  Eigen::Matrix2cd gate_test;
  gate_test << 1, 0, 0, std::complex<double>(0, 1);
  gate_test_func_2cd(gate, gate_test);
}

// T gate is ZPowGate with exponent of 0.25
TEST(GatesDefTest, T){
  const auto gate = ZPowGate().GetMatrix(0.25, 0);
  Eigen::Matrix2cd gate_test;
  gate_test << 1, 0, 0,
      std::complex<double>(1 / std::sqrt(2), 1 / std::sqrt(2));
  gate_test_func_2cd(gate, gate_test);
}

// RX gates are XPow gates with global shift of -0.5
TEST(GatesDefTest, RX) {
  for (auto const &angle : {0.123456, 5.4321}) {
    const auto gate = XPowGate().GetMatrix(angle/M_PI, -0.5);
    Eigen::Matrix2cd gate_test;
    gate_test << std::complex<double>(std::cos(angle / 2.), 0),
        std::complex<double>(0, -std::sin(angle / 2.)),
        std::complex<double>(0, -std::sin(angle / 2.)),
        std::complex<double>(std::cos(angle / 2.), 0);
    gate_test_func_2cd(gate, gate_test);
  }
}

// RY gates are YPow gates with global shift of -0.5
TEST(GatesDefTest, RY) {
  for (auto const &angle : {0.123456, 5.4321}) {
    const auto gate = YPowGate().GetMatrix(angle/M_PI, -0.5);
    Eigen::Matrix2cd gate_test;
    gate_test << std::complex<double>(std::cos(angle / 2.), 0),
        std::complex<double>(-std::sin(angle / 2.), 0),
        std::complex<double>(std::sin(angle / 2.), 0),
        std::complex<double>(std::cos(angle / 2.), 0);
    gate_test_func_2cd(gate, gate_test);
  }
}

// RZ gates are ZPow gates with global offset of -0.5
TEST(GatesDefTest, RZ) {
  for (auto const &angle : {0.123456, 5.4321}) {
    const auto gate = ZPowGate().GetMatrix(angle/M_PI, -0.5);
    Eigen::Matrix2cd gate_test;
    gate_test << std::exp(std::complex<double>(0, -1.0 * angle / 2.)), 0, 0,
        std::exp(std::complex<double>(0, angle / 2.));
    gate_test_func_2cd(gate, gate_test);
  }
}

// cirq XX gate is XXPowGate at exponent of 1
TEST(GatesDefTest, XX){
  const auto gate = XXPowGate().GetMatrix(1, 0);
  Eigen::Matrix4cd gate_test;
  gate_test << 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0;
  gate_test_func_4cd(gate, gate_test);
}

// cirq YY gate is YYPowGate at exponent of 1
TEST(GatesDefTest, YY){
  const auto gate = YYPowGate().GetMatrix(1, 0);
  Eigen::Matrix4cd gate_test;
  gate_test << 0, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0;
  gate_test_func_4cd(gate, gate_test);
}

// cirq ZZ gate is ZZPowGate at exponent of 1
TEST(GatesDefTest, ZZ){
  const auto gate = ZZPowGate().GetMatrix(1, 0);
  Eigen::Matrix4cd gate_test;
  gate_test << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1;
  gate_test_func_4cd(gate, gate_test);
}

// cirq CZ gate is CZPowGate at exponent of 1
TEST(GatesDefTest, CZ){
  const auto gate_1 = CZPowGate().GetMatrix(1, 0);
  const auto gate_2 = CZGate().GetMatrix();
  Eigen::Matrix4cd gate_test;
  gate_test << 1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, -1;
  for (const auto gate : {gate_1, gate_2}) {
    gate_test_func_4cd(gate, gate_test);
  }
}

// cirq CNot gate is CNotPowGate at exponent of 1
TEST(GatesDefTest, CNot){
  const auto gate_1 = CNotPowGate().GetMatrix(1, 0);
  const auto gate_2 = CNotGate().GetMatrix();
  Eigen::Matrix4cd gate_test;
  gate_test << 1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
    0, 0, 1, 0;
  for (const auto gate : {gate_1, gate_2}) {
    gate_test_func_4cd(gate, gate_test);
  }
}

// cirq Swap gate is SwapPowGate at exponent of 1
TEST(GatesDefTest, Swap){
  const auto gate_1 = SwapPowGate().GetMatrix(1, 0);
  const auto gate_2 = SwapGate().GetMatrix();
  Eigen::Matrix4cd gate_test;
  gate_test << 1, 0, 0, 0,
    0, 0, 1, 0,
    0, 1, 0, 0,
    0, 0, 0, 1;
  for (const auto gate : {gate_1, gate_2}) {
    gate_test_func_4cd(gate, gate_test);
  }
}

// cirq ISwap gate is ISwapPowGate at exponent of 1
TEST(GatesDefTest, ISwap){
  const auto gate_1 = ISwapPowGate().GetMatrix(1, 0);
  const auto gate_2 = ISwapGate().GetMatrix();
  Eigen::Matrix4cd gate_test;
  gate_test << 1, 0, 0, 0,
    0, 0, std::complex<double>(0, 1), 0,
    0, std::complex<double>(0, 1), 0, 0,
    0, 0, 0, 1;
  for (const auto gate : {gate_1, gate_2}) {
    gate_test_func_4cd(gate, gate_test);
  }
}

}  // namespace
}  // namespace tfq
