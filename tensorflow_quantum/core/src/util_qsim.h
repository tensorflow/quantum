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

#ifndef UTIL_QSIM_H_
#define UTIL_QSIM_H_

#include <cstdint>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

namespace tfq {

// Custom FOR loop struct to use TF threadpool instead of native
// qsim OpenMP or serial FOR implementations.
struct QsimFor {
  tensorflow::OpKernelContext* context;
  QsimFor(tensorflow::OpKernelContext* cxt) { context = cxt; }

  template <typename Function, typename... Args>
  void Run(uint64_t size, Function&& func, Args&&... args) const {
    auto worker_f = [&func, &args...](int64_t start, int64_t end) {
      for (uint64_t i = start; i < end; i++) {
        // First two arguments in RUN appear to be unused.
        std::forward<Function>(func)(-10, -10, i, std::forward<Args>(args)...);
      }
    };
    // estimated number of cpu cycles needed for one unit of work.
    // TODO(mbbrough): figure out this number from here:
    //   https://github.com/quantumlib/qsim/issues/147
    const int cycle_estimate = 500;
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        size, cycle_estimate, worker_f);
  }

  template <typename Function, typename Op, typename... Args>
  typename Op::result_type RunReduce(unsigned num_threads, uint64_t size,
                                     Function&& func, Op&& op,
                                     Args&&... args) const {
    // TODO(mbbrough): implement the rest of this for Expectation functions.
    return 0;
  }
};

}  // namespace tfq

#endif  // UTIL_QSIM_H_