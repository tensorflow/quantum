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

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow_quantum/core/qsim/state_space.h"

namespace tfq {
namespace qsim {

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

void sample_state(const StateSpace& space, const int m,
                  std::vector<uint64_t>* samples) {
  // An alternate would be to use:
  // tensorflow/core/lib/random/distribution_sampler.h which would have a
  // runtime of:
  // O(2 ** num_qubits + m) and additional mem O(2 ** num_qubits + m)
  // This method has (which is good because memory is expensive to get):
  // O(2 ** num_qubits + m * log(m)) and additional mem O(m)
  // Note: random samples in samples will appear in order.
  if (m == 0) {
    return;
  }
  tensorflow::random::PhiloxRandom philox(std::rand());
  tensorflow::random::SimplePhilox gen(&philox);

  double cdf_so_far = 0.0;
  std::vector<float> random_vals(m, 0.0);
  samples->reserve(m);
  for (int i = 0; i < m; i++) {
    random_vals[i] = gen.RandFloat();
  }
  std::sort(random_vals.begin(), random_vals.end());

  int j = 0;
  for (uint64_t i = 0; i < space.GetDimension(); i++) {
    const std::complex<float> f_amp = space.GetAmpl(i);
    const std::complex<double> d_amp = std::complex<double>(
        static_cast<double>(f_amp.real()), static_cast<double>(f_amp.imag()));
    cdf_so_far += std::norm(d_amp);
    while (random_vals[j] < cdf_so_far && j < m) {
      samples->push_back(i);
      j++;
    }
  }

  // Safety measure in case of state norm underflow.
  // Likely to not have huge impact.
  while (j < m) {
    samples->push_back(samples->at(samples->size() - 1));
    j++;
  }
}

}  // namespace qsim
}  // namespace tfq
