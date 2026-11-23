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

#ifndef TFQ_CORE_OPS_PARSE_CONTEXT
#define TFQ_CORE_OPS_PARSE_CONTEXT

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/proto/program.pb.h"

// Legacy helper for existing construct_f and tfq_resolve_parameters_op sites
// (new code should use ParallelForRangeStatus or ParallelForStatus).
#define NESTED_FN_STATUS_SYNC(global_status, local_status, global_lock) \
  if (TF_PREDICT_FALSE(!local_status.ok())) {                           \
    global_lock.Lock();                                                 \
    global_status = local_status;                                       \
    global_lock.Unlock();                                               \
    return;                                                             \
  }

namespace tfq {

// Runs `fn(start, end)` across CPU worker threads using `schedule` (int64_t
// cost_per_unit or ThreadPool::SchedulingParams) and returns OkStatus or the
// first non-OK status recorded across workers (sibling shards run to
// completion). Requires a valid `context` with initialized CPU worker threads.
template <typename Schedule, typename Fn>
inline absl::Status ParallelForRangeStatus(tensorflow::OpKernelContext* context,
                                           const int64_t total,
                                           const Schedule& schedule,
                                           const Fn& fn) {
  if (total <= 0) {
    return absl::OkStatus();
  }
  absl::Status status = absl::OkStatus();
  absl::Mutex mu;
  context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
      total, schedule, [&](int64_t start, int64_t end) {
        absl::Status local = fn(start, end);
        if (TF_PREDICT_FALSE(!local.ok())) {
          absl::MutexLock lock(&mu);
          if (status.ok()) {
            status = std::move(local);
          }
        }
      });
  return status;
}

// Runs `fn(i)` for `i` in `[0, total)` across CPU worker threads, exiting each
// shard on its first error and returning OkStatus or the first non-OK status
// recorded across workers. Requires a valid `context` with initialized CPU
// worker threads.
template <typename Fn>
inline absl::Status ParallelForStatus(tensorflow::OpKernelContext* context,
                                      const int64_t total,
                                      const int64_t cost_per_unit,
                                      const Fn& fn) {
  return ParallelForRangeStatus(
      context, total, cost_per_unit,
      [&](int64_t start, int64_t end) -> absl::Status {
        for (int64_t i = start; i < end; i++) {
          absl::Status local = fn(i);
          if (TF_PREDICT_FALSE(!local.ok())) {
            return local;
          }
        }
        return absl::OkStatus();
      });
}

// Simplest Program proto parsing
tensorflow::Status ParsePrograms(tensorflow::OpKernelContext* context,
                                 const std::string& input_name,
                                 std::vector<tfq::proto::Program>* programs);

// Simplest Program proto parsing in 2D.
tensorflow::Status ParsePrograms2D(
    tensorflow::OpKernelContext* context, const std::string& input_name,
    std::vector<std::vector<tfq::proto::Program>>* programs);

// Parses a vector of programs along with another vector of programs to append
tensorflow::Status GetProgramsAndProgramsToAppend(
    tensorflow::OpKernelContext* context,
    std::vector<tfq::proto::Program>* programs,
    std::vector<tfq::proto::Program>* programs_to_append);

// A parameter map is a mapping from the name of the parameter to the index in
// the input parameter value tensor (for gradient computations) and the value
// of the parameter (for forward computation).
typedef absl::flat_hash_map<std::string, std::pair<int, float>> SymbolMap;

// Parses Cirq Program protos out of the 'circuit_specs' input Tensor. Also
// resolves the QubitIds inside of the Program. Optionally will resolve the
// QubitIds found in programs into PauliSums such that they are consistent
// and correct with the original programs.
tensorflow::Status GetProgramsAndNumQubits(
    tensorflow::OpKernelContext* context,
    std::vector<tfq::proto::Program>* programs, std::vector<int>* num_qubits,
    std::vector<std::vector<tfq::proto::PauliSum>>* p_sums = nullptr,
    bool swap_endianness = false);

// Parses Cirq Program protos out of the 'circuit_specs' input Tensor. Also
// resolves the QubitIds inside of the Program. This override also parses and
// resolves other_programs. Ensuring all qubits found in programs[i] are also
// found in all programs[i][j] for all j.
tensorflow::Status GetProgramsAndNumQubits(
    tensorflow::OpKernelContext* context,
    std::vector<tfq::proto::Program>* programs, std::vector<int>* num_qubits,
    std::vector<std::vector<tfq::proto::Program>>* other_programs);

// Parses PauliSum protos out of the 'pauli_sums' input tensor. Note this
// function does NOT resolve QubitID's as any paulisum needs a reference
// program to "discover" all of the active qubits and define the ordering.
tensorflow::Status GetPauliSums(
    tensorflow::OpKernelContext* context,
    std::vector<std::vector<tfq::proto::PauliSum>>* p_sums);

// Parses the input context to construct the SymbolMaps for the entire batch.
// The two input Tensors are expected to be of size:
//
// symbol_names : [max_num_symbols]
// symbol_values: [batch_size, max_num_symbols]
//
// and the returns 'maps' is of size [batch_size], where each map contains all
// of the input symbols and their associated value.
tensorflow::Status GetSymbolMaps(tensorflow::OpKernelContext* context,
                                 std::vector<SymbolMap>* maps);

// Upper bound on the number of samples accepted for any single circuit or
// pauli sum. The per-thread work split in BalanceTrajectory and in the noisy
// ops' ComputeSmall uses int arithmetic such as
// (num_samples + num_threads - 1) / num_threads, which overflows when
// num_samples is near INT_MAX; the resulting negative chunk sizes make the
// worker threads skip sampling and leave the output tensor uninitialized.
// 1e7 is far above practical shot counts while leaving more than 200x
// headroom to INT_MAX for any thread count.
inline constexpr int kMaxNumSamples = 10000000;

// Parses the number of samples from the 'num_samples' input tensor. Every
// element must be in the range [1, kMaxNumSamples].
tensorflow::Status GetNumSamples(
    tensorflow::OpKernelContext* context,
    std::vector<std::vector<int>>* parsed_num_samples);

// Parses the 'num_samples' input tensor when it is expected to only
//   contain one element. The element must be in the range
//   [0, kMaxNumSamples]; 0 is legal because the sampling ops return an empty
//   result for it.
tensorflow::Status GetIndividualSample(tensorflow::OpKernelContext* context,
                                       int* n_samples);

// Parses the downstream gradients tensor. Used by adjoint op.
tensorflow::Status GetPrevGrads(
    tensorflow::OpKernelContext* context,
    std::vector<std::vector<float>>* parsed_prev_grads);

}  // namespace tfq

#endif  // TFQ_CORE_OPS_PARSE_CONTEXT
