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

#ifndef TFQ_CORE_QSIM_MATRIX_H_
#define TFQ_CORE_QSIM_MATRIX_H_

#include <iostream>
#include <vector>

#include "tensorflow_quantum/core/src/gates_def.h"

namespace tfq {

// Routines for 2x2 complex matrices.
// Matrices are arrays of floating-point numbers.
// There are no checks for validity of arguments.
// We do not care about performance here.

template <typename Array2>
inline void Matrix2SetZero(Array2& mat) {
  for (unsigned i = 0; i < 8; ++i) {
    mat[i] = 0;
  }
}

template <typename Array2>
inline void Matrix2SetId(Array2& mat) {
  Matrix2SetZero(mat);

  mat[0] = 1;
  mat[6] = 1;
}

template <typename Array1, typename Array2>
inline void Matrix2Set(const Array1& u, Array2& mat) {
  for (unsigned i = 0; i < 8; ++i) {
    mat[i] = u[i];
  }
}

// Multiply two 2x2 matrices.
template <typename Array1, typename Array2>
inline void Matrix2Multiply(const Array1& u, Array2& mat) {
  typename Array1::value_type mat0[8];
  for (unsigned i = 0; i < 8; ++i) {
    mat0[i] = mat[i];
  }

  for (unsigned i = 0; i < 2; ++i) {
    for (unsigned j = 0; j < 2; ++j) {
      typename Array1::value_type tr = 0;
      typename Array1::value_type ti = 0;

      for (unsigned k = 0; k < 2; ++k) {
        auto mr0 = mat0[4 * k + 2 * j + 0];
        auto mi0 = mat0[4 * k + 2 * j + 1];

        auto uik = &u[4 * i + 2 * k];

        tr += uik[0] * mr0 - uik[1] * mi0;
        ti += uik[0] * mi0 + uik[1] * mr0;
      }

      mat[4 * i + 2 * j + 0] = tr;
      mat[4 * i + 2 * j + 1] = ti;
    }
  }
}

// Routines for 4x4 complex matrices.
// Matrices are arrays of floating-point numbers.
// There are no checks for validity of arguments.
// We do not care about performance here.

template <typename Array2>
inline void Matrix4SetZero(Array2& mat) {
  for (unsigned i = 0; i < 32; ++i) {
    mat[i] = 0;
  }
}

template <typename Array2>
inline void Matrix4SetId(Array2& mat) {
  Matrix4SetZero(mat);

  mat[0] = 1;
  mat[10] = 1;
  mat[20] = 1;
  mat[30] = 1;
}

template <typename Array1, typename Array2>
inline void Matrix4Set(const Array1& u, Array2& mat) {
  for (unsigned i = 0; i < 32; ++i) {
    mat[i] = u[i];
  }
}

// Multiply 4x4 matrix by one qubit matrix corresponding to qubit 1.
// First arg is 2x2 matrix, second arg is 4x4 matrix.
// In this function, qubit order is taken to be big-endian. See #936
template <typename Array1, typename Array2>
inline void Matrix4Multiply20(const Array1& u, Array2& mat) {
  auto u00 = &u[0];
  auto u01 = &u[2];
  auto u10 = &u[4];
  auto u11 = &u[6];

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 2; ++j) {
      auto mr0 = mat[16 * j + 0 + 2 * i];
      auto mi0 = mat[16 * j + 1 + 2 * i];
      auto mr1 = mat[16 * j + 8 + 2 * i];
      auto mi1 = mat[16 * j + 9 + 2 * i];

      mat[16 * j + 0 + 2 * i] =
          u00[0] * mr0 - u00[1] * mi0 + u01[0] * mr1 - u01[1] * mi1;
      mat[16 * j + 1 + 2 * i] =
          u00[0] * mi0 + u00[1] * mr0 + u01[0] * mi1 + u01[1] * mr1;
      mat[16 * j + 8 + 2 * i] =
          u10[0] * mr0 - u10[1] * mi0 + u11[0] * mr1 - u11[1] * mi1;
      mat[16 * j + 9 + 2 * i] =
          u10[0] * mi0 + u10[1] * mr0 + u11[0] * mi1 + u11[1] * mr1;
    }
  }
}

// Multiply 4x4 matrix by one qubit matrix corresponding to qubit 0.
// First arg is 2x2 matrix, second arg is 4x4 matrix.
// In this function, qubit order is taken to be big-endian. See #936
template <typename Array1, typename Array2>
inline void Matrix4Multiply21(const Array1& u, Array2& mat) {
  auto u00 = &u[0];
  auto u01 = &u[2];
  auto u10 = &u[4];
  auto u11 = &u[6];

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 2; ++j) {
      auto mr0 = mat[8 * j + 0 + 2 * i];
      auto mi0 = mat[8 * j + 1 + 2 * i];
      auto mr1 = mat[8 * j + 16 + 2 * i];
      auto mi1 = mat[8 * j + 17 + 2 * i];

      mat[8 * j + 0 + 2 * i] =
          u00[0] * mr0 - u00[1] * mi0 + u01[0] * mr1 - u01[1] * mi1;
      mat[8 * j + 1 + 2 * i] =
          u00[0] * mi0 + u00[1] * mr0 + u01[0] * mi1 + u01[1] * mr1;
      mat[8 * j + 16 + 2 * i] =
          u10[0] * mr0 - u10[1] * mi0 + u11[0] * mr1 - u11[1] * mi1;
      mat[8 * j + 17 + 2 * i] =
          u10[0] * mi0 + u10[1] * mr0 + u11[0] * mi1 + u11[1] * mr1;
    }
  }
}

// Multiply two 4x4 matrices.
template <typename Array1, typename Array2>
inline void Matrix4Multiply(const Array1& u, Array2& mat) {
  typename Array1::value_type mat0[32];
  for (unsigned i = 0; i < 32; ++i) {
    mat0[i] = mat[i];
  }

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      typename Array1::value_type tr = 0;
      typename Array1::value_type ti = 0;

      for (unsigned k = 0; k < 4; ++k) {
        auto mr0 = mat0[8 * k + 2 * j + 0];
        auto mi0 = mat0[8 * k + 2 * j + 1];

        auto uik = &u[8 * i + 2 * k];

        tr += uik[0] * mr0 - uik[1] * mi0;
        ti += uik[0] * mi0 + uik[1] * mr0;
      }

      mat[8 * i + 2 * j + 0] = tr;
      mat[8 * i + 2 * j + 1] = ti;
    }
  }
}

// Calculate 4x4 fused gate matrix.
template <typename Array2>
inline void CalcMatrix4(unsigned q0, unsigned q1,
                        const std::vector<Gate*>& gates, Array2& mat) {
  Matrix4SetId(mat);

  for (auto pgate : gates) {
    if (pgate->num_qubits == 1) {
      if (pgate->qubits[0] == q0) {
        Matrix4Multiply20(pgate->matrix, mat);
      } else if (pgate->qubits[0] == q1) {
        Matrix4Multiply21(pgate->matrix, mat);
      }
    } else {
      Matrix4Multiply(pgate->matrix, mat);
    }
  }
}

// Permute 4x4 matrix to switch between two qubits.
template <typename Array2>
static void Matrix4Permute(Array2& mat) {
  std::swap(mat[2], mat[4]);
  std::swap(mat[3], mat[5]);
  std::swap(mat[8], mat[16]);
  std::swap(mat[9], mat[17]);
  std::swap(mat[10], mat[20]);
  std::swap(mat[11], mat[21]);
  std::swap(mat[12], mat[18]);
  std::swap(mat[13], mat[19]);
  std::swap(mat[14], mat[22]);
  std::swap(mat[15], mat[23]);
  std::swap(mat[26], mat[28]);
  std::swap(mat[27], mat[29]);
}

}  // namespace tfq

#endif  // TFQ_CORE_QSIM_MATRIX_H_
