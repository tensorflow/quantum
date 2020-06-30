// Copyright 2019 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef UTIL_QSIM_H_
#define UTIL_QSIM_H_

#include <cstdint>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

namespace tfq {

// Custom FOR loop struct to use TF threadpool instead of native
// qsim OpenMP or serial FOR implementations.
struct TFQQsimFor {
  static tensorflow::OpKernelContext* context;
  TFQQsimFor(tensorflow::OpKernelContext* cxt) {
    context = cxt;
  }

  template <typename Function, typename... Args>
  static void Run(unsigned num_threads, uint64_t size, Function&& func,
                  Args&&... args) {

    auto worker_f = [&func, &args...] (int64_t start, int64_t end) {
      for(uint64_t i=start;i<end;i++){
        // First two arguments in RUN appear to be unused.
        std::forward<Function>(func)(-10, -10, i, std::forward<Args>(args)...);
      }
    };
    // estimated number of cpu cycles needed for one unit of work.
    const int cycle_estimate = 500;
    //context->device()
    //    ->tensorflow_cpu_worker_threads()
    //    ->workers->ParallelFor(size, cycle_estimate, worker_f);
  }

  template <typename Function, typename Op, typename... Args>
  static typename Op::result_type RunReduce(unsigned num_threads, uint64_t size,
                                            Function&& func, Op&& op,
                                            Args&&... args) {
    return 0;
    /*if (num_threads == 0) return typename Op::result_type();

    std::vector<typename Op::result_type> partial_results(num_threads, 0);

    thread::Bundle bundle;
    for (int index = 0; index < num_threads; ++index) {
      typename Op::result_type* output = &partial_results[index];
      bundle.Add([index, num_threads, size, output, &func, &op, &args...] {
        unsigned n = num_threads;
        unsigned m = index;

        uint64_t i0 = size * m / n;
        uint64_t i1 = size * (m + 1) / n;

        typename Op::result_type partial_result = 0;

        for (uint64_t i = i0; i < i1; ++i) {
          partial_result = std::forward<Op>(op)(
              partial_result, std::forward<Function>(func)(
                                  n, m, i, std::forward<Args>(args)...));
        }

        *output = partial_result;
      });
    }
    bundle.JoinAll();

    typename Op::result_type result = partial_results[0];

    for (unsigned i = 1; i < num_threads; ++i) {
      result = op(result, partial_results[i]);
    }

    return result;
    */
  }
};

}  // namespace tfq

#endif  // UTIL_QSIM_H_