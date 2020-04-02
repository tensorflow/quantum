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

#include "gtest/gtest.h"

namespace tfq {
namespace qsim {
namespace {

TEST(Util, ComputeParity) {
  // Check all parities for all two qubit states
  uint64_t sample_01;
  absl::flat_hash_set<unsigned int> parity_();
  absl::flat_hash_set<unsigned int> parity_0({0});
  absl::flat_hash_set<unsigned int> parity_1({1});
  absl::flat_hash_set<unsigned int> parity_01({0, 1});
  // |00> <--> 0
  sample_01 = 0;
  ASSERT_EQ(ComputeParity(parity_, sample_01), 1);
  ASSERT_EQ(ComputeParity(parity_0, sample_01), 1);
  ASSERT_EQ(ComputeParity(parity_1, sample_01), 1);
  ASSERT_EQ(ComputeParity(parity_01, sample_01), 1);
  // |01> <--> 1
  sample_01 = 1;
  ASSERT_EQ(ComputeParity(parity_, sample_01), 1);
  ASSERT_EQ(ComputeParity(parity_0, sample_01), -1);
  ASSERT_EQ(ComputeParity(parity_1, sample_01), 1);
  ASSERT_EQ(ComputeParity(parity_01, sample_01), -1);
  // |10> <--> 2
  sample_01 = 2;
  ASSERT_EQ(ComputeParity(parity_, sample_01), 1);
  ASSERT_EQ(ComputeParity(parity_0, sample_01), 1);
  ASSERT_EQ(ComputeParity(parity_1, sample_01), -1);
  ASSERT_EQ(ComputeParity(parity_01, sample_01), -1);
  // |11> <--> 3
  sample_01 = 3;
  ASSERT_EQ(ComputeParity(parity_, sample_01), 1);
  ASSERT_EQ(ComputeParity(parity_0, sample_01), -1);
  ASSERT_EQ(ComputeParity(parity_1, sample_01), -1);
  ASSERT_EQ(ComputeParity(parity_01, sample_01), 1);

  // Check all parities for a three qubit state
  // |101> <--> 5
  uint64_t sample_012(5);
  absl::flat_hash_set<unsigned int> parity_2();
  absl::flat_hash_set<unsigned int> parity_02({0, 2});
  absl::flat_hash_set<unsigned int> parity_12({1, 2});
  absl::flat_hash_set<unsigned int> parity_012({0, 1, 2});
  ASSERT_EQ(ComputeParity(parity_, sample_012), 1);
  ASSERT_EQ(ComputeParity(parity_0, sample_012), -1);
  ASSERT_EQ(ComputeParity(parity_1, sample_012), 1);
  ASSERT_EQ(ComputeParity(parity_01, sample_012), -1);
  ASSERT_EQ(ComputeParity(parity_2, sample_012), -1);
  ASSERT_EQ(ComputeParity(parity_02, sample_012), 1);
  ASSERT_EQ(ComputeParity(parity_12, sample_012), -1);
  ASSERT_EQ(ComputeParity(parity_012, sample_012), 1);
  
  // Check some parities for a six qubit state
  // |010000> <--> 26
  uint64_t sample_six(26);
  absl::flat_hash_set<unsigned int> parity_012345({0, 1, 2, 3, 4, 5});
  absl::flat_hash_set<unsigned int> parity_134({1, 3, 4});
  absl::flat_hash_set<unsigned int> parity_025({0, 2, 5});
  ASSERT_EQ(ComputeParity(parity_, sample_six), 1);
  ASSERT_EQ(ComputeParity(parity_0, sample_six), 1);
  ASSERT_EQ(ComputeParity(parity_1, sample_six), 1);
  ASSERT_EQ(ComputeParity(parity_01, sample_six), 1);
  ASSERT_EQ(ComputeParity(parity_2, sample_six), 1);
  ASSERT_EQ(ComputeParity(parity_02, sample_six), 1);
  ASSERT_EQ(ComputeParity(parity_12, sample_six), -1);
  ASSERT_EQ(ComputeParity(parity_012, sample_six), -1);
  ASSERT_EQ(ComputeParity(parity_012345, sample_six), -1);
  ASSERT_EQ(ComputeParity(parity_134, sample_six), -1);
  ASSERT_EQ(ComputeParity(parity_025, sample_six), 1);
}

}  // namespace
}  // namespace qsim
}  // namespace tfq
