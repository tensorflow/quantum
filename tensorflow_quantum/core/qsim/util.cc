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

#include <cstddef>
#include <cstdlib>

namespace tfq {
namespace qsim_old {

void* _aligned_malloc(size_t size) {
  // choose 64 bit alignment in case we ever introduce avx 512 support.
  const size_t al = 64;
  void* initial_mem = malloc(size + al);
  void* aligned_mem = reinterpret_cast<void*>(
      (reinterpret_cast<size_t>(initial_mem) & ~(al - 1)) + al);
  *(reinterpret_cast<void**>(aligned_mem) - 1) = initial_mem;
  return aligned_mem;
}

void _aligned_free(void* ptr) { free(*(reinterpret_cast<void**>(ptr) - 1)); }

}  // namespace qsim_old
}  // namespace tfq
