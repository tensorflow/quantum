/* Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_quantum/core/qsim/q_state.h"

#ifdef __AVX2__
#include "tensorflow_quantum/core/qsim/simulator2_avx.h"
#endif

#if __SSE4_1__
#include "tensorflow_quantum/core/qsim/simulator2_sse.h"
#endif

#include <google/protobuf/text_format.h>

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "cirq/google/api/v2/program.pb.h"
#include "gtest/gtest.h"
#include "tensorflow_quantum/core/qsim/q_state.h"
#include "tensorflow_quantum/core/qsim/simulator.h"
#include "tensorflow_quantum/core/qsim/simulator2_slow.h"
#include "tensorflow_quantum/core/src/circuit.h"
#include "tensorflow_quantum/core/src/circuit_parser.h"

namespace tfq {
namespace qsim {
namespace {

using ::cirq::google::api::v2::Program;

void BasicTest(QState* state) {
  const std::string text = R"(
    circuit {
      scheduling_strategy: MOMENT_BY_MOMENT
      moments {
        operations {
          gate { id: "HP" }
          qubits { id: "0" }
          args {
              key: "global_shift"
              value: {arg_value: {float_value: 0.0}}
          }
          args {
              key: "exponent"
              value: {arg_value: {float_value: 1.0}}
          }
          args {
              key: "exponent_scalar"
              value: {arg_value: {float_value: 1.0}}
          }
        }
        operations {
          gate { id: "HP" }
          qubits { id: "1" }
          args {
              key: "global_shift"
              value: {arg_value: {float_value: 0.0}}
          }
          args {
              key: "exponent"
              value: {arg_value: {float_value: 1.0}}
          }
          args {
              key: "exponent_scalar"
              value: {arg_value: {float_value: 1.0}}
          }
        }
        operations {
          gate { id: "HP" }
          qubits { id: "2" }
          args {
              key: "global_shift"
              value: {arg_value: {float_value: 0.0}}
          }
          args {
              key: "exponent"
              value: {arg_value: {float_value: 1.0}}
          }
          args {
              key: "exponent_scalar"
              value: {arg_value: {float_value: 1.0}}
          }
        }
      }
    }
  )";

  Program program;
  google::protobuf::TextFormat::ParseFromString(text, &program);

  Circuit circuit;
  tensorflow::Status status = CircuitFromProgram(program, 3, &circuit);
  ASSERT_TRUE(status.ok());

  status = state->Update(circuit);
  ASSERT_TRUE(status.ok());

  for (int i = 0; i < 8; i++) {
    auto amp = state->GetAmplitude(i);
    EXPECT_NEAR(amp.real(), 0.3535, 0.001);
    EXPECT_NEAR(amp.imag(), 0.0, 0.001);
    state->SetAmplitude(i, std::complex<float>(0, 0));
    amp = state->GetAmplitude(i);
    EXPECT_NEAR(amp.real(), 0, 0.001);
    EXPECT_NEAR(amp.imag(), 0, 0.001);
  }
}

TEST(QStateTest, BasicSlow) {
  const int num_qubits = 3;
  std::unique_ptr<Simulator> simulator =
      absl::make_unique<Simulator2Slow>(num_qubits, 1);
  QState state(std::move(simulator), num_qubits);
  BasicTest(&state);
}

#ifdef __AVX2__
TEST(QStateTest, BasicAVX) {
  const int num_qubits = 3;
  std::unique_ptr<Simulator> simulator =
      absl::make_unique<Simulator2AVX>(num_qubits, 1);
  QState state(std::move(simulator), num_qubits);
  BasicTest(&state);
}
#endif

#if __SSE4_1__
TEST(QStateTest, BasicSSE) {
  const int num_qubits = 3;
  std::unique_ptr<Simulator> simulator =
      absl::make_unique<Simulator2SSE>(num_qubits, 1);
  QState state(std::move(simulator), num_qubits);
  BasicTest(&state);
}
#endif

}  // namespace
}  // namespace qsim
}  // namespace tfq
