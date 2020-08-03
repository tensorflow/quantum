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

#include "tensorflow_quantum/core/qsim/util.h"

#include <vector>

#include "gtest/gtest.h"

namespace tfq {
namespace qsim_old {
namespace {

TEST(Util, ComputeBitmask) {
  const std::vector<unsigned int> parity_;
  const std::vector<unsigned int> parity_0({0});
  const std::vector<unsigned int> parity_1({1});
  const std::vector<unsigned int> parity_01({0, 1});
  const std::vector<unsigned int> parity_2({2});
  const std::vector<unsigned int> parity_02({0, 2});
  const std::vector<unsigned int> parity_12({1, 2});
  const std::vector<unsigned int> parity_012({0, 1, 2});
  ASSERT_EQ(ComputeBitmask(parity_), 0);
  ASSERT_EQ(ComputeBitmask(parity_0), 1);
  ASSERT_EQ(ComputeBitmask(parity_1), 2);
  ASSERT_EQ(ComputeBitmask(parity_01), 3);
  ASSERT_EQ(ComputeBitmask(parity_2), 4);
  ASSERT_EQ(ComputeBitmask(parity_02), 5);
  ASSERT_EQ(ComputeBitmask(parity_12), 6);
  ASSERT_EQ(ComputeBitmask(parity_012), 7);
}

TEST(Util, ComputeParity) {
  // Check all parities for all two qubit states
  uint64_t sample_01;
  // |00> <--> 0
  sample_01 = 0;
  ASSERT_EQ(ComputeParity(0, sample_01), 1);
  ASSERT_EQ(ComputeParity(1, sample_01), 1);
  ASSERT_EQ(ComputeParity(2, sample_01), 1);
  ASSERT_EQ(ComputeParity(3, sample_01), 1);
  // |01> <--> 1
  sample_01 = 1;
  ASSERT_EQ(ComputeParity(0, sample_01), 1);
  ASSERT_EQ(ComputeParity(1, sample_01), -1);
  ASSERT_EQ(ComputeParity(2, sample_01), 1);
  ASSERT_EQ(ComputeParity(3, sample_01), -1);
  // |10> <--> 2
  sample_01 = 2;
  ASSERT_EQ(ComputeParity(0, sample_01), 1);
  ASSERT_EQ(ComputeParity(1, sample_01), 1);
  ASSERT_EQ(ComputeParity(2, sample_01), -1);
  ASSERT_EQ(ComputeParity(3, sample_01), -1);
  // |11> <--> 3
  sample_01 = 3;
  ASSERT_EQ(ComputeParity(0, sample_01), 1);
  ASSERT_EQ(ComputeParity(1, sample_01), -1);
  ASSERT_EQ(ComputeParity(2, sample_01), -1);
  ASSERT_EQ(ComputeParity(3, sample_01), 1);

  // Check all parities for a three qubit state
  // |101> <--> 5
  uint64_t sample_012(5);
  ASSERT_EQ(ComputeParity(0, sample_012), 1);
  ASSERT_EQ(ComputeParity(1, sample_012), -1);
  ASSERT_EQ(ComputeParity(2, sample_012), 1);
  ASSERT_EQ(ComputeParity(3, sample_012), -1);
  ASSERT_EQ(ComputeParity(4, sample_012), -1);
  ASSERT_EQ(ComputeParity(5, sample_012), 1);
  ASSERT_EQ(ComputeParity(6, sample_012), -1);
  ASSERT_EQ(ComputeParity(7, sample_012), 1);
  // |100> <--> 4
  sample_012 = 4;
  ASSERT_EQ(ComputeParity(0, sample_012), 1);
  ASSERT_EQ(ComputeParity(1, sample_012), 1);
  ASSERT_EQ(ComputeParity(2, sample_012), 1);
  ASSERT_EQ(ComputeParity(3, sample_012), 1);
  ASSERT_EQ(ComputeParity(4, sample_012), -1);
  ASSERT_EQ(ComputeParity(5, sample_012), -1);
  ASSERT_EQ(ComputeParity(6, sample_012), -1);
  ASSERT_EQ(ComputeParity(7, sample_012), -1);

  // Check some parities for a six qubit state
  // |011000> <--> 24
  uint64_t sample_six(24);
  ASSERT_EQ(ComputeParity(0, sample_six), 1);
  ASSERT_EQ(ComputeParity(1, sample_six), 1);
  ASSERT_EQ(ComputeParity(2, sample_six), 1);
  ASSERT_EQ(ComputeParity(3, sample_six), 1);
  ASSERT_EQ(ComputeParity(4, sample_six), 1);
  ASSERT_EQ(ComputeParity(5, sample_six), 1);
  ASSERT_EQ(ComputeParity(6, sample_six), 1);
  ASSERT_EQ(ComputeParity(7, sample_six), 1);
  ASSERT_EQ(ComputeParity(63, sample_six), 1);
  ASSERT_EQ(ComputeParity(25, sample_six), 1);
  ASSERT_EQ(ComputeParity(37, sample_six), 1);
  ASSERT_EQ(ComputeParity(16, sample_six), -1);
}

}  // namespace
}  // namespace qsim_old
}  // namespace tfq
