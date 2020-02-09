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

#include <cstdlib>

#include "gtest/gtest.h"

namespace tfq {
namespace {

TEST(GatesDefTest, GateConstructors) {
  // Empty gate constructor
  Gate gate0q;
  ASSERT_EQ(gate0q.time, 0);
  ASSERT_EQ(gate0q.num_qubits, 0);

  // One-qubit gate constructor
  const unsigned int time1q = 256;
  const unsigned int qubits1q = 53;
  const std::array<float, 8> matrix1q{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
  Gate gate1q(time1q, qubits1q, matrix1q);
  ASSERT_EQ(gate1q.time, time1q);
  ASSERT_EQ(gate1q.num_qubits, 1);
  ASSERT_EQ(gate1q.qubits[0], qubits1q);
  for (int i = 0; i < 8; i++) {
    ASSERT_EQ(gate1q.matrix[i], matrix1q[i]);
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
  ASSERT_EQ(gate2q.time, time2q);
  ASSERT_EQ(gate2q.num_qubits, 2);
  ASSERT_EQ(gate2q.qubits[0], qubits2q1);
  ASSERT_EQ(gate2q.qubits[1], qubits2q2);
  for (int i = 0; i < 32; i++) {
    ASSERT_EQ(gate2q.matrix[i], matrix2q[i]);
  }
}

TEST(GatesDefTest, GateEquality) {
  // Empty gate
  Gate test_gate_0q, real_gate_0q;

  test_gate_0q.time = real_gate_0q.time + 1;
  ASSERT_NE(test_gate_0q, real_gate_0q);
  test_gate_0q.time = real_gate_0q.time;

  test_gate_0q.num_qubits = real_gate_0q.num_qubits + 1;
  ASSERT_NE(test_gate_0q, real_gate_0q);
  test_gate_0q.num_qubits = real_gate_0q.num_qubits;

  ASSERT_EQ(test_gate_0q, real_gate_0q);

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

}  // namespace
}  // namespace tfq
