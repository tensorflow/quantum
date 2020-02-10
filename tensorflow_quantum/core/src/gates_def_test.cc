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

}  // namespace
}  // namespace tfq
