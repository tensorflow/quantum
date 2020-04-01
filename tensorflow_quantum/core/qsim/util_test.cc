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
  // Check parities for |11> <--> 3
  uint64_t sample_01 = 3;
  absl::flat_hash_set<unsigned int> parity_();
  absl::flat_hash_set<unsigned int> parity_0({0});
  absl::flat_hash_set<unsigned int> parity_1({1});
  absl::flat_hash_set<unsigned int> parity_01({0, 1});
  ASSERT_EQ(ComputeParity(parity_, sample_01), 1);
  ASSERT_EQ(ComputeParity(parity_0, sample_01), -1);
  ASSERT_EQ(ComputeParity(parity_1, sample_01), -1);
  ASSERT_EQ(ComputeParity(parity_01, sample_01), 1);
}

}  // namespace
}  // namespace qsim
}  // namespace tfq
