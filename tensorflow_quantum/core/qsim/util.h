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

#ifndef TFQ_CORE_QSIM_UTIL_H_
#define TFQ_CORE_QSIM_UTIL_H_

#include <cstddef>

#include "absl/container/flat_hash_set.h"

namespace tfq {
namespace qsim {

// Workaround for std::aligned_alloc not working on C++11.
void* _aligned_malloc(size_t size);

// Workaround for std::alligned_alloc not working on C++11.
void _aligned_free(void* ptr);

// Given a set of qubit indices and an integer state sample, return the parity
// of that set of indices.  Uses the little-endian convention of qsim.
int ComputeParity(const absl::flat_hash_set<unsigned int>& parity_set,
                  const uint64_t sample);

}  // namespace qsim
}  // namespace tfq

#endif  // TFQ_CORE_QSIM_UTIL_H_
