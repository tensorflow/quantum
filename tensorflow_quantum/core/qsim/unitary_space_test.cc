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

#include "tensorflow_quantum/core/qsim/unitary_space.h"

#include <Eigen/Dense>
#include <complex>
#include <memory>
#include <unsupported/Eigen/KroneckerProduct>

#include "gtest/gtest.h"
#include "tensorflow_quantum/core/qsim/mux.h"
#include "tensorflow_quantum/core/src/circuit.h"

namespace tfq {
namespace qsim {
namespace {

TEST(UnitaryTest, GetSetEntry) {
  auto unitary = GetUnitarySpace(1, 1);
  unitary->CreateUnitary();
  unitary->SetEntry(0, 0, std::complex<float>(1.0, 2.0));
  unitary->SetEntry(0, 1, std::complex<float>(3.0, 4.0));
  unitary->SetEntry(1, 0, std::complex<float>(5.0, 6.0));
  unitary->SetEntry(1, 1, std::complex<float>(7.0, 8.0));

  EXPECT_EQ(unitary->GetEntry(0, 0), std::complex<float>(1.0, 2.0));
  EXPECT_EQ(unitary->GetEntry(0, 1), std::complex<float>(3.0, 4.0));
  EXPECT_EQ(unitary->GetEntry(1, 0), std::complex<float>(5.0, 6.0));
  EXPECT_EQ(unitary->GetEntry(1, 1), std::complex<float>(7.0, 8.0));
}

TEST(UnitaryTest, SetIdentity) {
  auto unitary = GetUnitarySpace(3, 1);
  unitary->CreateUnitary();
  unitary->SetIdentity();

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      if (i == j) {
        EXPECT_EQ(unitary->GetEntry(i, j), std::complex<float>(1.0, 0.0));
      } else {
        EXPECT_EQ(unitary->GetEntry(i, j), std::complex<float>(0.0, 0.0));
      }
    }
  }
}

TEST(UnitaryTest, Delete) {
  auto unitary = GetUnitarySpace(3, 1);
  unitary->CreateUnitary();
  EXPECT_TRUE(unitary->Valid());
  unitary->DeleteUnitary();
  EXPECT_FALSE(unitary->Valid());
}

TEST(UnitaryTest, Getters) {
  auto unitary = GetUnitarySpace(5, 2);
  EXPECT_EQ(unitary->GetNumQubits(), 5);
  EXPECT_EQ(unitary->GetNumThreads(), 2);
  EXPECT_EQ(unitary->GetType(), UnitarySpaceType::USLOW);
}

TEST(UnitaryTest, OneDimensionUnitary) {
  auto unitary = GetUnitarySpace(1, 1);
  unitary->CreateUnitary();
  unitary->SetIdentity();
  float H_gate[] = {0.707, 0, 0.707, 0, 0.707, 0, -0.707, 0};
  unitary->ApplyGate1(H_gate);
  EXPECT_EQ(unitary->GetEntry(0, 0), std::complex<float>(0.707, 0.0));
  EXPECT_EQ(unitary->GetEntry(0, 1), std::complex<float>(0.707, 0.0));
  EXPECT_EQ(unitary->GetEntry(1, 0), std::complex<float>(0.707, 0.0));
  EXPECT_EQ(unitary->GetEntry(1, 1), std::complex<float>(-0.707, 0.0));
  float full_matrix[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  unitary->SetIdentity();
  unitary->ApplyGate1(full_matrix);
  EXPECT_EQ(unitary->GetEntry(0, 0), std::complex<float>(1.0, 2.0));
  EXPECT_EQ(unitary->GetEntry(0, 1), std::complex<float>(3.0, 4.0));
  EXPECT_EQ(unitary->GetEntry(1, 0), std::complex<float>(5.0, 6.0));
  EXPECT_EQ(unitary->GetEntry(1, 1), std::complex<float>(7.0, 8.0));
  unitary->ApplyGate1(full_matrix);
  EXPECT_EQ(unitary->GetEntry(0, 0), std::complex<float>(-12.0, 42.0));
  EXPECT_EQ(unitary->GetEntry(0, 1), std::complex<float>(-16.0, 62.0));
  EXPECT_EQ(unitary->GetEntry(1, 0), std::complex<float>(-20.0, 98.0));
  EXPECT_EQ(unitary->GetEntry(1, 1), std::complex<float>(-24.0, 150.0));
}

TEST(UnitaryTest, SimpleTwoDimensionUnitary) {
  auto unitary = GetUnitarySpace(2, 1);
  unitary->CreateUnitary();
  unitary->SetIdentity();
  //clang-format off
  float full_gate[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                       9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                       17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                       25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0};
  //clang-format on
  unitary->SetIdentity();
  unitary->ApplyGate2(0, 1, full_gate);
  EXPECT_EQ(unitary->GetEntry(0, 0), std::complex<float>(1.0, 2.0));
  EXPECT_EQ(unitary->GetEntry(0, 1), std::complex<float>(3.0, 4.0));
  EXPECT_EQ(unitary->GetEntry(0, 2), std::complex<float>(5.0, 6.0));
  EXPECT_EQ(unitary->GetEntry(0, 3), std::complex<float>(7.0, 8.0));
  EXPECT_EQ(unitary->GetEntry(1, 0), std::complex<float>(9.0, 10.0));
  EXPECT_EQ(unitary->GetEntry(1, 1), std::complex<float>(11.0, 12.0));
  EXPECT_EQ(unitary->GetEntry(1, 2), std::complex<float>(13.0, 14.0));
  EXPECT_EQ(unitary->GetEntry(1, 3), std::complex<float>(15.0, 16.0));
  EXPECT_EQ(unitary->GetEntry(2, 0), std::complex<float>(17.0, 18.0));
  EXPECT_EQ(unitary->GetEntry(2, 1), std::complex<float>(19.0, 20.0));
  EXPECT_EQ(unitary->GetEntry(2, 2), std::complex<float>(21.0, 22.0));
  EXPECT_EQ(unitary->GetEntry(2, 3), std::complex<float>(23.0, 24.0));
  EXPECT_EQ(unitary->GetEntry(3, 0), std::complex<float>(25.0, 26.0));
  EXPECT_EQ(unitary->GetEntry(3, 1), std::complex<float>(27.0, 28.0));
  EXPECT_EQ(unitary->GetEntry(3, 2), std::complex<float>(29.0, 30.0));
  EXPECT_EQ(unitary->GetEntry(3, 3), std::complex<float>(31.0, 32.0));
  unitary->ApplyGate2(0, 1, full_gate);
  EXPECT_EQ(unitary->GetEntry(0, 0), std::complex<float>(-72.0, 644.0));
  EXPECT_EQ(unitary->GetEntry(0, 1), std::complex<float>(-80.0, 716.0));
  EXPECT_EQ(unitary->GetEntry(0, 2), std::complex<float>(-88.0, 788.0));
  EXPECT_EQ(unitary->GetEntry(0, 3), std::complex<float>(-96.0, 860.0));
  EXPECT_EQ(unitary->GetEntry(1, 0), std::complex<float>(-104.0, 1508.0));
  EXPECT_EQ(unitary->GetEntry(1, 1), std::complex<float>(-112.0, 1708.0));
  EXPECT_EQ(unitary->GetEntry(1, 2), std::complex<float>(-120.0, 1908.0));
  EXPECT_EQ(unitary->GetEntry(1, 3), std::complex<float>(-128.0, 2108.0));
  EXPECT_EQ(unitary->GetEntry(2, 0), std::complex<float>(-136.0, 2372.0));
  EXPECT_EQ(unitary->GetEntry(2, 1), std::complex<float>(-144.0, 2700.0));
  EXPECT_EQ(unitary->GetEntry(2, 2), std::complex<float>(-152.0, 3028.0));
  EXPECT_EQ(unitary->GetEntry(2, 3), std::complex<float>(-160.0, 3356.0));
  EXPECT_EQ(unitary->GetEntry(3, 0), std::complex<float>(-168.0, 3236.0));
  EXPECT_EQ(unitary->GetEntry(3, 1), std::complex<float>(-176.0, 3692.0));
  EXPECT_EQ(unitary->GetEntry(3, 2), std::complex<float>(-184.0, 4148.0));
  EXPECT_EQ(unitary->GetEntry(3, 3), std::complex<float>(-192.0, 4604.0));
}

TEST(UnitaryTest, ThreeDimensionUnitary01) {
  auto unitary = GetUnitarySpace(3, 1);
  unitary->CreateUnitary();
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      unitary->SetEntry(
          i, j, std::complex<float>(16 * i + 2 * j, 16 * i + 2 * j + 1));
    }
  }

  Eigen::MatrixXcf ref_u =
      Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic,
                               Eigen::Dynamic, Eigen::RowMajor>>(
          (std::complex<float>*)unitary->GetRawUnitary(), 8, 8);

  //clang-format off
  float full_gate[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                       9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                       17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                       25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0};
  //clang-format on
  Eigen::MatrixXcf ref_gate =
      Eigen::Map<Eigen::Matrix<std::complex<float>, 4, 4, Eigen::RowMajor>>(
          (std::complex<float>*)full_gate);
  Eigen::MatrixXcf up_gate = Eigen::MatrixXcf(2, 2);
  up_gate.setIdentity();
  up_gate = kroneckerProduct(ref_gate, up_gate).eval();
  up_gate = (up_gate * ref_u).eval();

  unitary->ApplyGate2(0, 1, full_gate);
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      EXPECT_EQ(unitary->GetEntry(i, j), up_gate(i, j));
    }
  }
}

TEST(UnitaryTest, ThreeDimensionUnitary12) {
  auto unitary = GetUnitarySpace(3, 1);
  unitary->CreateUnitary();
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      unitary->SetEntry(
          i, j, std::complex<float>(16 * i + 2 * j, 16 * i + 2 * j + 1));
    }
  }

  Eigen::MatrixXcf ref_u =
      Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic,
                               Eigen::Dynamic, Eigen::RowMajor>>(
          (std::complex<float>*)unitary->GetRawUnitary(), 8, 8);

  //clang-format off
  float full_gate[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                       9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                       17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                       25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0};
  //clang-format on
  Eigen::MatrixXcf ref_gate =
      Eigen::Map<Eigen::Matrix<std::complex<float>, 4, 4, Eigen::RowMajor>>(
          (std::complex<float>*)full_gate);
  Eigen::MatrixXcf up_gate = Eigen::MatrixXcf(2, 2);
  up_gate.setIdentity();
  up_gate = kroneckerProduct(up_gate, ref_gate).eval();
  up_gate = (up_gate * ref_u).eval();

  unitary->ApplyGate2(1, 2, full_gate);
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      EXPECT_EQ(unitary->GetEntry(i, j), up_gate(i, j));
    }
  }
}

TEST(UnitaryTest, ThreeDimensionUnitary02) {
  auto unitary = GetUnitarySpace(3, 1);
  unitary->CreateUnitary();
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      unitary->SetEntry(
          i, j, std::complex<float>(16 * i + 2 * j, 16 * i + 2 * j + 1));
    }
  }

  Eigen::MatrixXcf ref_u =
      Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic,
                               Eigen::Dynamic, Eigen::RowMajor>>(
          (std::complex<float>*)unitary->GetRawUnitary(), 8, 8);

  // clang-format off
  float full_gate[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0,
                       12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
                       23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0};
  float swap_gate[] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  // clang-format on
  Eigen::MatrixXcf ref_gate =
      Eigen::Map<Eigen::Matrix<std::complex<float>, 4, 4, Eigen::RowMajor>>(
          (std::complex<float>*)full_gate);
  Eigen::MatrixXcf swap01 =
      Eigen::Map<Eigen::Matrix<std::complex<float>, 4, 4, Eigen::RowMajor>>(
          (std::complex<float>*)swap_gate);

  Eigen::MatrixXcf up_gate = Eigen::MatrixXcf(2, 2);
  up_gate.setIdentity();

  Eigen::MatrixXcf swap12 = kroneckerProduct(up_gate, swap01).eval();

  up_gate = kroneckerProduct(ref_gate, up_gate).eval();
  up_gate = (swap12 * up_gate * swap12 * ref_u).eval();

  unitary->ApplyGate2(0, 2, full_gate);
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      EXPECT_EQ(unitary->GetEntry(i, j), up_gate(i, j));
    }
  }
}

TEST(UnitaryGate, UpdateSimpleOneQubit) {
  const std::array<float, 8> matrix_i = {1.0, 2.0, 3.0, 4.0,
                                         5.0, 6.0, 7.0, 8.0};
  Gate gate_i(0, 0, matrix_i);
  std::vector<Gate> gates{gate_i};
  const Circuit circuit(1, gates);

  auto unitary = GetUnitarySpace(1, 1);
  unitary->CreateUnitary();
  unitary->SetIdentity();
  unitary->Update(circuit);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_EQ(unitary->GetEntry(i, j).real(), matrix_i[4 * i + 2 * j]);
      EXPECT_EQ(unitary->GetEntry(i, j).imag(), matrix_i[4 * i + 2 * j + 1]);
    }
  }
}

TEST(UnitaryGate, UpdateSimpleTwoQubit) {
  // clang-format off
  const std::array<float, 32> matrix_i = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0,
      12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
      23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0};
  // clang-format on
  Gate gate_i(0, 0, 1, matrix_i);
  std::vector<Gate> gates{gate_i};
  const Circuit circuit(2, gates);

  auto unitary = GetUnitarySpace(2, 1);
  unitary->CreateUnitary();
  unitary->SetIdentity();
  unitary->Update(circuit);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(unitary->GetEntry(i, j).real(), matrix_i[8 * i + 2 * j]);
      EXPECT_EQ(unitary->GetEntry(i, j).imag(), matrix_i[8 * i + 2 * j + 1]);
    }
  }
}

}  // namespace
}  // namespace qsim
}  // namespace tfq
