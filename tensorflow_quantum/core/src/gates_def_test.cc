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

#include "tensorflow_quantum/core/src/circuit.h"

#include <cstdlib>

#include "gtest/gtest.h"

namespace tfq {
namespace {

float RandomFloat() {
  return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}

void GetTestGate1q(Gate* test_gate) {
  std::array<float, 8> matrix;
  std::generate(matrix.begin(), matrix.begin() + 8, RandomFloat);
  *test_gate = Gate(18, 3, matrix);
}

void GetTestGate2q(Gate* test_gate) {
  std::array<float, 32> matrix;
  std::generate(matrix.begin(), matrix.begin() + 32, RandomFloat);
  *test_gate = Gate(19, 3, 4, matrix);
}

TEST(GatesDefTest, GateConstructors) {
  // Empty gate constructor
  Gate gate0q();
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
  const std::array<float, 32> matrix2q{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31};
  Gate gate2q(time2q, qubits2q1, qubits2q2, matrix2q);
  ASSERT_EQ(gate2q.time, time2q);
  ASSERT_EQ(gate2q.num_qubits, 2);
  ASSERT_EQ(gate2q.qubits[0], qubits2q1);
  ASSERT_EQ(gate2q.qubits[1], qubits2q2);
  for (int i = 0; i < 32; i++) {
    ASSERT_EQ(gate2q.matrix[i], matrix2q[i]);
  }
}

TEST(CircuitTest, Gate0q) {
  Gate test_gate, true_gate;

  // check individual element equality manually
  ASSERT_EQ(test_gate.time, true_gate.time);
  ASSERT_EQ(test_gate.num_qubits, true_gate.num_qubits);

  // check equality operator overload
  test_gate.time = true_gate.time + 1;
  ASSERT_NE(test_gate, true_gate);
  test_gate.time = true_gate.time;

  test_gate.num_qubits = true_gate.num_qubits + 1;
  ASSERT_NE(test_gate, true_gate);
  test_gate.num_qubits = true_gate.num_qubits;

  ASSERT_EQ(test_gate, true_gate);
}

TEST(CircuitTest, Gate1q) {
  Gate test_gate;
  GetTestGate1q(&test_gate);
  Gate true_gate;
  true_gate = test_gate;

  // check individual element equality manually
  ASSERT_EQ(test_gate.time, true_gate.time);
  ASSERT_EQ(test_gate.num_qubits, true_gate.num_qubits);
  ASSERT_EQ(test_gate.qubits[0], true_gate.qubits[0]);
  for (int i = 0; i < 8; i++) {
    ASSERT_EQ(test_gate.matrix[i], true_gate.matrix[i]);
  }

  // check equality operator overload
  test_gate.time = true_gate.time + 1;
  ASSERT_NE(test_gate, true_gate);
  test_gate.time = true_gate.time;

  test_gate.num_qubits = true_gate.num_qubits + 1;
  ASSERT_NE(test_gate, true_gate);
  test_gate.num_qubits = true_gate.num_qubits;

  test_gate.qubits[0] = true_gate.qubits[0] + 1;
  ASSERT_NE(test_gate, true_gate);
  test_gate.qubits[0] = true_gate.qubits[0];

  test_gate.matrix[7] = true_gate.matrix[7] + 1;
  ASSERT_NE(test_gate, true_gate);
  test_gate.matrix[7] = true_gate.matrix[7];

  ASSERT_EQ(test_gate, true_gate);
}

TEST(CircuitTest, Gate2q) {
  Gate test_gate;
  GetTestGate2q(&test_gate);
  Gate true_gate;
  true_gate = test_gate;

  // check individual element equality manually
  ASSERT_EQ(test_gate.time, true_gate.time);
  ASSERT_EQ(test_gate.num_qubits, true_gate.num_qubits);
  ASSERT_EQ(test_gate.qubits[0], true_gate.qubits[0]);
  ASSERT_EQ(test_gate.qubits[1], true_gate.qubits[1]);
  for (int i = 0; i < 32; i++) {
    ASSERT_EQ(test_gate.matrix[i], true_gate.matrix[i]);
  }

  // check equality operator overload
  test_gate.time = true_gate.time + 1;
  ASSERT_NE(test_gate, true_gate);
  test_gate.time = true_gate.time;

  test_gate.num_qubits = true_gate.num_qubits + 1;
  ASSERT_NE(test_gate, true_gate);
  test_gate.num_qubits = true_gate.num_qubits;

  test_gate.qubits[0] = true_gate.qubits[0] + 1;
  ASSERT_NE(test_gate, true_gate);
  test_gate.qubits[0] = true_gate.qubits[0];

  test_gate.qubits[1] = true_gate.qubits[1] + 1;
  ASSERT_NE(test_gate, true_gate);
  test_gate.qubits[1] = true_gate.qubits[1];

  test_gate.matrix[31] = true_gate.matrix[31] + 1;
  ASSERT_NE(test_gate, true_gate);
  test_gate.matrix[31] = true_gate.matrix[31];

  ASSERT_EQ(test_gate, true_gate);
}


}  // namespace
}  // namespace tfq
