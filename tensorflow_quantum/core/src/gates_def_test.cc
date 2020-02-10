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

// ============================================================================
// GateBuilder implementation tests.
// ============================================================================

TEST(GatesDefTest, XPow){
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
  builder.Build(time, locations, arg_map, &test_gate);
  ASSERT_EQ(test_gate, real_gate);

  // // RX gates are XPow gates with global shift of -0.5
  // const float angle = 0.123456;
  // std::array<float, 8> matrix_rot{std::cos(angle / 2.), 0, 0, -std::sin(angle / 2.), 0, -std::sin(angle / 2.), std::cos(angle / 2.), 0};
  // Gate real_gate_rot(time, qubit, matrix_rot);
  // absl::flat_hash_map<std::string, float> arg_map_rot;
  // arg_map["global_shift"] = 0.0;
  // arg_map["exponent"] = 1.0;
  // arg_map["exponent_scalar"] = 1.0;
  // Gate test_gate_rot;
  // builder.Build(time, locations, arg_map_rot, &test_gate_rot);
  // ASSERT_EQ(test_gate_rot, real_gate_rot);

  // for (auto const &angle : {0.123456, 5.4321}) {
  //   const auto gate = XPowGate().GetMatrix(angle/M_PI, -0.5);
  //   Eigen::Matrix2cd gate_test;
  //   gate_test << std::complex<double>(std::cos(angle / 2.), 0),
  //       std::complex<double>(0, -std::sin(angle / 2.)),
  //       std::complex<double>(0, -std::sin(angle / 2.)),
  //       std::complex<double>(std::cos(angle / 2.), 0);
  //   gate_test_func_2cd(gate, gate_test);
  // }
}

TEST(GatesDefTest, YPow){
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
  builder.Build(time, locations, arg_map, &test_gate);
  ASSERT_EQ(test_gate, real_gate);
}

TEST(GatesDefTest, ZPow){
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
  builder.Build(time, locations, arg_map, &test_gate);
  ASSERT_EQ(test_gate, real_gate);
}

TEST(GatesDefTest, HPow){
  HPowGateBuilder builder;
  const unsigned int time{3};
  const unsigned int qubit{53};
  std::vector<unsigned int> locations;
  locations.push_back(qubit);

  // cirq H gate is HPowGate at exponent of 1.
  std::array<float, 8> matrix{1/std::sqrt(2), 0, -1/std::sqrt(2), 0, 1/std::sqrt(2), 0, 1/std::sqrt(2), 0};
  Gate real_gate(time, qubit, matrix);
  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = 0.0;
  arg_map["exponent"] = 1.0;
  arg_map["exponent_scalar"] = 1.0;
  Gate test_gate;
  builder.Build(time, locations, arg_map, &test_gate);
  ASSERT_EQ(test_gate, real_gate);
}

TEST(GatesDefTest, IdentityGate){
  IGateBuilder builder;
  const unsigned int time{3};
  const unsigned int qubit{53};
  std::vector<unsigned int> locations;
  locations.push_back(qubit);

  std::array<float, 8> matrix{1, 0, 0, 0, 0, 0, 1, 0};
  Gate real_gate(time, qubit, matrix);
  absl::flat_hash_map<std::string, float> arg_map;
  Gate test_gate;
  builder.Build(time, locations, arg_map, &test_gate);
  ASSERT_EQ(test_gate, real_gate);
}

// TEST(GatesDefTest, CNotGate){
//   Eigen::Matrix4cd gate = CNotGate().GetMatrix();
//   Eigen::Matrix4cd gate_test;
//   gate_test << 1, 0, 0, 0,
//     0, 1, 0, 0,
//     0, 0, 0, 1,
//     0, 0, 1, 0;
//   gate_test_func_4cd(gate, gate_test);
// }

// // S gate is ZPowGate with exponent of 0.5
// TEST(GatesDefTest, S){
//   const auto gate = ZPowGate().GetMatrix(0.5, 0);
//   Eigen::Matrix2cd gate_test;
//   gate_test << 1, 0, 0, std::complex<double>(0, 1);
//   gate_test_func_2cd(gate, gate_test);
// }

// // T gate is ZPowGate with exponent of 0.25
// TEST(GatesDefTest, T){
//   const auto gate = ZPowGate().GetMatrix(0.25, 0);
//   Eigen::Matrix2cd gate_test;
//   gate_test << 1, 0, 0,
//       std::complex<double>(1 / std::sqrt(2), 1 / std::sqrt(2));
//   gate_test_func_2cd(gate, gate_test);
// }

// // RY gates are YPow gates with global shift of -0.5
// TEST(GatesDefTest, RY) {
//   for (auto const &angle : {0.123456, 5.4321}) {
//     const auto gate = YPowGate().GetMatrix(angle/M_PI, -0.5);
//     Eigen::Matrix2cd gate_test;
//     gate_test << std::complex<double>(std::cos(angle / 2.), 0),
//         std::complex<double>(-std::sin(angle / 2.), 0),
//         std::complex<double>(std::sin(angle / 2.), 0),
//         std::complex<double>(std::cos(angle / 2.), 0);
//     gate_test_func_2cd(gate, gate_test);
//   }
// }

// // RZ gates are ZPow gates with global offset of -0.5
// TEST(GatesDefTest, RZ) {
//   for (auto const &angle : {0.123456, 5.4321}) {
//     const auto gate = ZPowGate().GetMatrix(angle/M_PI, -0.5);
//     Eigen::Matrix2cd gate_test;
//     gate_test << std::exp(std::complex<double>(0, -1.0 * angle / 2.)), 0, 0,
//         std::exp(std::complex<double>(0, angle / 2.));
//     gate_test_func_2cd(gate, gate_test);
//   }
// }

// // cirq XX gate is XXPowGate at exponent of 1
// TEST(GatesDefTest, XX){
//   const auto gate = XXPowGate().GetMatrix(1, 0);
//   Eigen::Matrix4cd gate_test;
//   gate_test << 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0;
//   gate_test_func_4cd(gate, gate_test);
// }

// // cirq YY gate is YYPowGate at exponent of 1
// TEST(GatesDefTest, YY){
//   const auto gate = YYPowGate().GetMatrix(1, 0);
//   Eigen::Matrix4cd gate_test;
//   gate_test << 0, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0;
//   gate_test_func_4cd(gate, gate_test);
// }

// // cirq ZZ gate is ZZPowGate at exponent of 1
// TEST(GatesDefTest, ZZ){
//   const auto gate = ZZPowGate().GetMatrix(1, 0);
//   Eigen::Matrix4cd gate_test;
//   gate_test << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1;
//   gate_test_func_4cd(gate, gate_test);
// }

// // cirq CZ gate is CZPowGate at exponent of 1
// TEST(GatesDefTest, CZ){
//   const auto gate_1 = CZPowGate().GetMatrix(1, 0);
//   const auto gate_2 = CZGate().GetMatrix();
//   Eigen::Matrix4cd gate_test;
//   gate_test << 1, 0, 0, 0,
//     0, 1, 0, 0,
//     0, 0, 1, 0,
//     0, 0, 0, -1;
//   for (const auto gate : {gate_1, gate_2}) {
//     gate_test_func_4cd(gate, gate_test);
//   }
// }

// // cirq CNot gate is CNotPowGate at exponent of 1
// TEST(GatesDefTest, CNot){
//   const auto gate_1 = CNotPowGate().GetMatrix(1, 0);
//   const auto gate_2 = CNotGate().GetMatrix();
//   Eigen::Matrix4cd gate_test;
//   gate_test << 1, 0, 0, 0,
//     0, 1, 0, 0,
//     0, 0, 0, 1,
//     0, 0, 1, 0;
//   for (const auto gate : {gate_1, gate_2}) {
//     gate_test_func_4cd(gate, gate_test);
//   }
// }

// // cirq Swap gate is SwapPowGate at exponent of 1
// TEST(GatesDefTest, Swap){
//   const auto gate_1 = SwapPowGate().GetMatrix(1, 0);
//   const auto gate_2 = SwapGate().GetMatrix();
//   Eigen::Matrix4cd gate_test;
//   gate_test << 1, 0, 0, 0,
//     0, 0, 1, 0,
//     0, 1, 0, 0,
//     0, 0, 0, 1;
//   for (const auto gate : {gate_1, gate_2}) {
//     gate_test_func_4cd(gate, gate_test);
//   }
// }

// // cirq ISwap gate is ISwapPowGate at exponent of 1
// TEST(GatesDefTest, ISwap){
//   const auto gate_1 = ISwapPowGate().GetMatrix(1, 0);
//   const auto gate_2 = ISwapGate().GetMatrix();
//   Eigen::Matrix4cd gate_test;
//   gate_test << 1, 0, 0, 0,
//     0, 0, std::complex<double>(0, 1), 0,
//     0, std::complex<double>(0, 1), 0, 0,
//     0, 0, 0, 1;
//   for (const auto gate : {gate_1, gate_2}) {
//     gate_test_func_4cd(gate, gate_test);
//   }
// }

}  // namespace
}  // namespace tfq
