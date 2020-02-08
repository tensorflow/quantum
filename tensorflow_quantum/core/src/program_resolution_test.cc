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

#include "tensorflow_quantum/core/src/program_resolution.h"

#include <google/protobuf/text_format.h>

#include <string>

#include "absl/container/flat_hash_map.h"
#include "cirq/google/api/v2/program.pb.h"
#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status.h"

namespace tfq {
namespace {

using cirq::google::api::v2::Program;
using tensorflow::Status;

TEST(ProgramResolutionTest, ResolveQubitIds) {
  const std::string text = R"(
    circuit {
      moments {
        operations {
          qubits {
            id: "0_0"
          }
          qubits {
            id: "1_0"
          }
        }
      }
      moments {
        operations {
          qubits {
            id: "0_0"
          }
          qubits {
            id: "0_1"
          }
        }
      }
    }
  )";

  const std::string text_empty = R"(
      circuit {
      }
    )";

  Program program, empty_program;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(text, &program));
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(text_empty,
                                                            &empty_program));

  EXPECT_TRUE(ResolveQubitIds(&program).ok());
  EXPECT_TRUE(ResolveQubitIds(&empty_program).ok());

  EXPECT_EQ(program.circuit().moments(0).operations(0).qubits(0).id(), "0");
  EXPECT_EQ(program.circuit().moments(0).operations(0).qubits(1).id(), "2");
  EXPECT_EQ(program.circuit().moments(1).operations(0).qubits(0).id(), "0");
  EXPECT_EQ(program.circuit().moments(1).operations(0).qubits(1).id(), "1");
}

TEST(ProgramResolutionTest, ResolveSymbols) {
  const std::string text = R"(
    circuit {
      scheduling_strategy: MOMENT_BY_MOMENT
      moments {
        operations {
          args {
            key: "exponent"
            value {
              symbol: "v1"
            }
          }
        }
      }
    }
  )";

  Program program;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(text, &program));

  const absl::flat_hash_map<std::string, std::pair<int, double>> param_map = {
      {"v1", {0, 1.0}}};

  EXPECT_TRUE(ResolveSymbols(param_map, &program).ok());
  EXPECT_EQ(program.circuit()
                .moments(0)
                .operations(0)
                .args()
                .at("exponent")
                .arg_value()
                .float_value(),
            1.0);
}

}  // namespace
}  // namespace tfq
