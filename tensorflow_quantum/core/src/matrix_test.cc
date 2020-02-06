/* Copyright 2020 The TensorFlow Quantum authors. All Rights Reserved.

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

#include "tensorflow_quantum/core/src/matrix.h"

#include <stdlib.h>

#include <complex>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

namespace tfq {

namespace {

float RandomFloat() {
  float random = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  return random;
}

TEST(MatrixTest, Matrix2SetZero) {
  std::array<float, 8> matrix;
  std::generate(begin(matrix), end(matrix), RandomFloat);
  Matrix2SetZero(matrix);
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(matrix[i], 0.0);
  }
}

TEST(MatrixTest, Matrix2SetId) {
  std::array<float, 8> matrix;
  std::generate(begin(matrix), end(matrix), RandomFloat);
  Matrix2SetId(matrix);
  for (int i = 0; i < 8; i++) {
    if (i == 0 || i == 6) {
      EXPECT_EQ(matrix[i], 1.0);
      continue;
    }
    EXPECT_EQ(matrix[i], 0.0);
  }
}

TEST(MatrixTest, Matrix2Set) {
  std::array<float, 8> matrix;
  std::generate(begin(matrix), end(matrix), RandomFloat);
  std::array<float, 8> target{1, 2, 3, 4, 5, 6, 7, 8};

  Matrix2Set(target, matrix);
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(matrix[i], target[i]);
  }
}

TEST(MatrixTest, Matrix2Multiply) {
  std::array<float, 8> a, b;
  std::generate(begin(a), end(a), RandomFloat);
  std::generate(begin(b), end(b), RandomFloat);
  // Just use complex matmul on transformed versions of the above
  std::complex<float> c[2][2], d[2][2], f[2][2];
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      c[i][j] = std::complex<float>(a[4 * i + 2 * j], a[4 * i + 2 * j + 1]);
      d[i][j] = std::complex<float>(b[4 * i + 2 * j], b[4 * i + 2 * j + 1]);
    }
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        f[i][j] += c[i][k] * d[k][j];
      }
    }
  }

  Matrix2Multiply(a, b);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_NEAR(real(f[i][j]), b[4 * i + 2 * j], 1E-6);
      EXPECT_NEAR(imag(f[i][j]), b[4 * i + 2 * j + 1], 1E-6);
    }
  }
}

TEST(MatrixTest, Matrix4SetZero) {
  std::array<float, 32> matrix;
  std::generate(begin(matrix), end(matrix), RandomFloat);
  Matrix4SetZero(matrix);
  for (int i = 0; i < 32; i++) {
    EXPECT_EQ(matrix[i], 0.0);
  }
}

TEST(MatrixTest, Matrix4SetId) {
  std::array<float, 32> matrix;
  std::generate(begin(matrix), end(matrix), RandomFloat);
  Matrix4SetId(matrix);
  for (int i = 0; i < 32; i++) {
    if (i == 0 || i == 10 || i == 20 || i == 30) {
      EXPECT_EQ(matrix[i], 1.0);
      continue;
    }
    EXPECT_EQ(matrix[i], 0.0);
  }
}

TEST(MatrixTest, Matrix4Set) {
  std::array<float, 32> matrix;
  std::generate(begin(matrix), end(matrix), RandomFloat);
  std::array<float, 32> target;
  for (int i = 0; i < 32; i++) {
    target[i] = i;
  }

  Matrix4Set(target, matrix);
  for (int i = 0; i < 32; i++) {
    EXPECT_EQ(matrix[i], target[i]);
  }
}

TEST(MatrixTest, Matrix4Multiply21) {
  std::array<float, 8> a;
  std::array<float, 32> b;
  std::generate(begin(a), end(a), RandomFloat);
  std::generate(begin(b), end(b), RandomFloat);

  // Tensor up single-qubit matrix for ordinary complex matmul.
  // the matrix below represents C \otimes Id
  std::complex<float> c[4][4], d[4][4], f[4][4];
  c[0][0] = std::complex<float>(a[0], a[1]);
  c[1][1] = c[0][0];
  c[0][2] = std::complex<float>(a[2], a[3]);
  c[1][3] = c[0][2];
  c[2][0] = std::complex<float>(a[4], a[5]);
  c[3][1] = c[2][0];
  c[2][2] = std::complex<float>(a[6], a[7]);
  c[3][3] = c[2][2];

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        f[i][j] += c[i][k] *
                   std::complex<float>(b[8 * k + 2 * j], b[8 * k + 2 * j + 1]);
      }
    }
  }

  Matrix4Multiply21(a, b);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_NEAR(real(f[i][j]), b[8 * i + 2 * j], 1E-6);
      EXPECT_NEAR(imag(f[i][j]), b[8 * i + 2 * j + 1], 1E-6);
    }
  }
}

TEST(MatrixTest, Matrix4Multiply20) {
  std::array<float, 8> a;
  std::array<float, 32> b;
  std::generate(begin(a), end(a), RandomFloat);
  std::generate(begin(b), end(b), RandomFloat);

  // Use complex matmul on transformed versions of the above.
  // the matrix below represents Id \otimes C
  std::complex<float> c[4][4], d[4][4], f[4][4];
  c[0][0] = std::complex<float>(a[0], a[1]);
  c[0][1] = std::complex<float>(a[2], a[3]);
  c[1][0] = std::complex<float>(a[4], a[5]);
  c[1][1] = std::complex<float>(a[6], a[7]);
  c[2][2] = c[0][0];
  c[2][3] = c[0][1];
  c[3][2] = c[1][0];
  c[3][3] = c[1][1];

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        f[i][j] += c[i][k] *
                   std::complex<float>(b[8 * k + 2 * j], b[8 * k + 2 * j + 1]);
      }
    }
  }

  Matrix4Multiply20(a, b);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_NEAR(real(f[i][j]), b[8 * i + 2 * j], 1E-6);
      EXPECT_NEAR(imag(f[i][j]), b[8 * i + 2 * j + 1], 1E-6);
    }
  }
}

TEST(MatrixTest, Matrix4Multiply) {
  std::array<float, 32> a, b;
  std::generate(begin(a), end(a), RandomFloat);
  std::generate(begin(b), end(b), RandomFloat);
  // Just use complex matmul on transformed versions of the above
  std::complex<float> c[4][4], d[4][4], f[4][4];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = std::complex<float>(a[8 * i + 2 * j], a[8 * i + 2 * j + 1]);
      d[i][j] = std::complex<float>(b[8 * i + 2 * j], b[8 * i + 2 * j + 1]);
    }
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        f[i][j] += c[i][k] * d[k][j];
      }
    }
  }

  Matrix4Multiply(a, b);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_NEAR(real(f[i][j]), b[8 * i + 2 * j], 1E-6);
      EXPECT_NEAR(imag(f[i][j]), b[8 * i + 2 * j + 1], 1E-6);
    }
  }
}

}  // namespace
}  // namespace tfq
