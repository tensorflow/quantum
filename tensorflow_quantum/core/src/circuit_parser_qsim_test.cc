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

#include "tensorflow_quantum/core/src/circuit_parser_qsim.h"

#include <google/protobuf/text_format.h>

#include <string>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/gates_cirq.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "cirq/google/api/v2/program.pb.h"
#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status.h"

namespace tfq {
namespace {

// using ::cirq::google::api::v2::Program;
// using ::qsim::Cirq::GateCirq;

TEST(QsimCircuitParserTest, qsimthing) {
  cirq::google::api::v2::Program program_proto;
  qsim::Circuit<qsim::Cirq::GateCirq<float>> real_circuit;
  ASSERT_TRUE(1);
}

}  // namespace
}  // namespace tfq
