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

#include "tensorflow_quantum/core/qsim/state_space.h"

#include <cmath>
#include <complex>
#include <memory>

#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/qsim/mux.h"

#ifdef __AVX2__
tfq::qsim::StateSpaceType STATE_SPACE_TYPE = tfq::qsim::StateSpaceType::AVX;
#elif __SSE4_1__
tfq::qsim::StateSpaceType STATE_SPACE_TYPE = tfq::qsim::StateSpaceType::SSE;
#else
tfq::qsim::StateSpaceType STATE_SPACE_TYPE = tfq::qsim::StateSpaceType::SLOW;
#endif

namespace tfq {
namespace qsim {
namespace {

TEST(StateSpaceTest, Initialization) {
  uint64_t num_qubits = 4;
  uint64_t num_threads = 5;
  auto state =
      std::unique_ptr<StateSpace>(GetStateSpace(num_qubits, num_threads));
  ASSERT_FALSE(state->Valid());
  ASSERT_FALSE(state->GetRawState());
  ASSERT_EQ(state->GetDimension(), 1 << num_qubits);
  ASSERT_EQ(state->GetNumQubits(), num_qubits);
  ASSERT_EQ(state->GetNumThreads(), num_threads);

  state->CreateState();
  ASSERT_TRUE(state->Valid());
  ASSERT_TRUE(state->GetRawState());
  ASSERT_EQ(state->GetDimension(), 1 << num_qubits);
  ASSERT_EQ(state->GetNumQubits(), num_qubits);
  ASSERT_EQ(state->GetNumThreads(), num_threads);

  ASSERT_EQ(state->GetType(), STATE_SPACE_TYPE);

  state->DeleteState();
  ASSERT_FALSE(state->Valid());
  ASSERT_FALSE(state->GetRawState());
}

TEST(StateSpaceTest, CloneTest) {
  auto state = std::unique_ptr<StateSpace>(GetStateSpace(5, 3));
  auto state_clone = std::unique_ptr<StateSpace>(state->Clone());

  ASSERT_EQ(state->GetDimension(), state_clone->GetDimension());
  ASSERT_EQ(state->GetNumQubits(), state_clone->GetNumQubits());
  ASSERT_EQ(state->GetNumThreads(), state_clone->GetNumThreads());
}

TEST(StateSpaceTest, Amplitudes) {
  auto state = std::unique_ptr<StateSpace>(GetStateSpace(2, 1));
  state->CreateState();

  std::complex<float> ampl_00(0.1, 0.5);
  std::complex<float> ampl_01(0.2, 0.6);
  std::complex<float> ampl_10(0.3, 0.7);
  std::complex<float> ampl_11(0.4, 0.8);

  state->SetAmpl(0, ampl_00);
  state->SetAmpl(1, ampl_01);
  state->SetAmpl(2, ampl_10);
  state->SetAmpl(3, ampl_11);

  ASSERT_EQ(state->GetAmpl(0), ampl_00);
  ASSERT_EQ(state->GetAmpl(1), ampl_01);
  ASSERT_EQ(state->GetAmpl(2), ampl_10);
  ASSERT_EQ(state->GetAmpl(3), ampl_11);

  state->SetStateZero();
  ASSERT_EQ(state->GetAmpl(0), std::complex<float>(1, 0));
  ASSERT_EQ(state->GetAmpl(1), std::complex<float>(0, 0));
  ASSERT_EQ(state->GetAmpl(2), std::complex<float>(0, 0));
  ASSERT_EQ(state->GetAmpl(3), std::complex<float>(0, 0));
}

TEST(StateSpaceTest, CopyFromGetRealInnerProduct) {
  auto state = std::unique_ptr<StateSpace>(GetStateSpace(12, 1));
  state->CreateState();
  for (uint64_t i = 0; i < state->GetDimension(); i++) {
    state->SetAmpl(i, std::complex<float>(i, i));
  }

  auto state_clone = std::unique_ptr<StateSpace>(state->Clone());
  state_clone->CreateState();
  state_clone->CopyFrom(*state);

  for (uint64_t i = 0; i < state->GetDimension(); i++) {
    ASSERT_EQ(state->GetAmpl(i), state_clone->GetAmpl(i));
  }

  double real_inner_product = 0.0;
  for (uint64_t i = 0; i < state->GetDimension(); i++) {
    real_inner_product += 2 * i * i;
  }

  EXPECT_NEAR(state->GetRealInnerProduct(*state_clone), real_inner_product,
              1E-2);
}

TEST(StateSpaceTest, ApplyGate1) {
  auto state = std::unique_ptr<StateSpace>(GetStateSpace(5, 1));
  state->CreateState();
  const float matrix[] = {1.0 / std::sqrt(2), 0.0, 1.0 / std::sqrt(2), 0.0,
                          1.0 / std::sqrt(2), 0.0, -1.0 / std::sqrt(2), 0.0};
  state->SetStateZero();
  state->SetAmpl(0, std::complex<float>(0.0, 0.0));
  state->SetAmpl(1, std::complex<float>(1.0, 0.0));
  switch (state->GetType()) {
    case StateSpaceType::AVX:
      ASSERT_EQ(
          state->ApplyGate1(matrix),
          tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                             "AVX simulator doesn't support small circuits."));
      break;
    case StateSpaceType::SLOW:
      state->ApplyGate1(matrix);
      ASSERT_EQ(state->GetAmpl(0), std::complex<float>(1/std::sqrt(2), 0.0));
      ASSERT_EQ(state->GetAmpl(1), std::complex<float>(-1/std::sqrt(2), 0.0));
      break;
    case StateSpaceType::SSE:
      ASSERT_EQ(
          state->ApplyGate1(matrix),
          tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                             "SSE simulator doesn't support small circuits."));
      break;
  }
}

TEST(StateSpaceTest, ApplyGate2) {
  auto state = std::unique_ptr<StateSpace>(GetStateSpace(6, 1));
  state->CreateState();
  const float matrix_ih[] = {1.0 / std::sqrt(2), 0.0, 1.0 / std::sqrt(2), 0.0, 0.0, 0.0, 0.0, 0.0,
                             1.0 / std::sqrt(2), 0.0, -1.0 / std::sqrt(2), 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 1.0 / std::sqrt(2), 0.0, 1.0 / std::sqrt(2), 0.0,
                             0.0, 0.0, 0.0, 0.0, 1.0 / std::sqrt(2), 0.0, -1.0 / std::sqrt(2), 0.0};
  const float matrix_cnot[] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  state->SetStateZero();
  state->SetAmpl(0, std::complex<float>(0.0, 0.0));
  state->SetAmpl(8, std::complex<float>(1.0, 0.0));
  state->ApplyGate2(3, 4, matrix_ih);
  ASSERT_EQ(state->GetAmpl(8), std::complex<float>(1/std::sqrt(2), 0.0));
  ASSERT_EQ(state->GetAmpl(9), std::complex<float>(1/std::sqrt(2), 0.0));
  // state->ApplyGate2(3, 4, matrix_cnot);
  // ASSERT_EQ(state->GetAmpl(8), std::complex<float>(1/std::sqrt(2), 0.0));
  // ASSERT_EQ(state->GetAmpl(11), std::complex<float>(1/std::sqrt(2), 0.0));
}

TEST(StateSpaceTest, SampleStateOneSample) {
  auto equal = std::unique_ptr<StateSpace>(GetStateSpace(1, 1));
  equal->CreateState();
  equal->SetAmpl(0, std::complex<float>(1.0, 0.));
  equal->SetAmpl(1, std::complex<float>(0.0, 0.));

  std::vector<uint64_t> samples;
  equal->SampleState(1, &samples);
  ASSERT_EQ(samples.size(), 1);
}

TEST(StateSpaceTest, SampleStateZeroSamples) {
  auto equal = std::unique_ptr<StateSpace>(GetStateSpace(1, 1));
  equal->CreateState();
  equal->SetAmpl(0, std::complex<float>(1.0, 0.));
  equal->SetAmpl(1, std::complex<float>(0.0, 0.));

  std::vector<uint64_t> samples;
  equal->SampleState(0, &samples);
  ASSERT_EQ(samples.size(), 0);
}

TEST(StateSpaceTest, SampleStateEqual) {
  auto equal = std::unique_ptr<StateSpace>(GetStateSpace(1, 1));
  equal->CreateState();
  equal->SetAmpl(0, std::complex<float>(0.707, 0.));
  equal->SetAmpl(1, std::complex<float>(0.707, 0.));

  std::vector<uint64_t> samples;
  const int m = 100000;
  equal->SampleState(m, &samples);

  float num_ones = 0.0;
  for (int i = 0; i < m; i++) {
    if (samples[i]) {
      num_ones++;
    }
  }
  ASSERT_EQ(samples.size(), m);
  EXPECT_NEAR(num_ones / static_cast<float>(m), 0.5, 1E-2);
}

TEST(StateSpaceTest, SampleStateSkew) {
  auto skew = std::unique_ptr<StateSpace>(GetStateSpace(1, 1));
  skew->CreateState();

  std::vector<float> rots = {0.1, 0.3, 0.5, 0.7, 0.9};
  for (int t = 0; t < 5; t++) {
    float z_amp = std::sqrt(rots[t]);
    float o_amp = std::sqrt(1.0 - rots[t]);
    skew->SetAmpl(0, std::complex<float>(z_amp, 0.));
    skew->SetAmpl(1, std::complex<float>(o_amp, 0.));

    std::vector<uint64_t> samples;
    const int m = 100000;
    skew->SampleState(m, &samples);
    float num_z = 0.0;
    for (int i = 0; i < m; i++) {
      if (samples[i] == 0) {
        num_z++;
      }
    }
    ASSERT_EQ(samples.size(), m);
    EXPECT_NEAR(num_z / static_cast<float>(m), rots[t], 1E-2);
  }
}

TEST(StateSpaceTest, SampleStateComplexDist) {
  auto state = std::unique_ptr<StateSpace>(GetStateSpace(3, 1));
  state->CreateState();

  std::vector<float> probs = {0.05, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2};
  for (int i = 0; i < 8; i++) {
    state->SetAmpl(i, std::complex<float>(std::sqrt(probs[i]), 0.0));
  }

  std::vector<uint64_t> samples;
  const int m = 100000;
  state->SampleState(m, &samples);

  std::vector<float> measure_probs = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int i = 0; i < m; i++) {
    measure_probs[samples[i]] += 1.0;
  }
  for (int i = 0; i < 8; i++) {
    EXPECT_NEAR(measure_probs[i] / static_cast<float>(m), probs[i], 1E-2);
  }
  ASSERT_EQ(samples.size(), m);
}

}  // namespace
}  // namespace qsim
}  // namespace tfq
