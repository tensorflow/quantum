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

#include <bitset>
#include <cstdint>
#include <vector>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/fuser.h"
#include "../qsim/lib/gate_appl.h"
#include "../qsim/lib/gates_cirq.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/src/circuit_parser_qsim.h"

namespace tfq {

typedef absl::flat_hash_map<std::string, std::pair<int, float>> SymbolMap;
typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;

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
    //   https://github.com/quantumlib/qsim/issues/147
    const int cycle_estimate = 100;
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        size, cycle_estimate, worker_f);
  }

  uint64_t GetIndex0(uint64_t size, unsigned thread_id) const {
    unsigned int num_threads = context->device()
                                   ->tensorflow_cpu_worker_threads()
                                   ->workers->NumThreads();
    return size * thread_id / num_threads;
  }

  uint64_t GetIndex1(uint64_t size, unsigned thread_id) const {
    unsigned int num_threads = context->device()
                                   ->tensorflow_cpu_worker_threads()
                                   ->workers->NumThreads();
    return size * (thread_id + 1) / num_threads;
  }

  template <typename Function, typename Op, typename... Args>
  std::vector<typename Op::result_type> RunReduceP(uint64_t size,
                                                   Function&& func, Op&& op,
                                                   Args&&... args) const {
    unsigned int num_threads = context->device()
                                   ->tensorflow_cpu_worker_threads()
                                   ->workers->NumThreads();

    std::vector<typename Op::result_type> partial_results(num_threads, 0);

    std::function<void(int64_t, int64_t)> fn =
        [this, &size, &num_threads, &partial_results, &func, &op, &args...](
            int64_t start, int64_t end) {
          // Here we expect end = start + 1 because block_size = 1.
          uint64_t i0 = GetIndex0(size, start);
          uint64_t i1 = GetIndex1(size, start);

          typename Op::result_type partial_result = 0;

          for (uint64_t i = i0; i < i1; i++) {
            partial_result =
                op(partial_result, func(num_threads, start, i, args...));
          }

          partial_results[start] = partial_result;
        };

    // block_size = 1.
    tensorflow::thread::ThreadPool::SchedulingParams scheduling_params(
        tensorflow::thread::ThreadPool::SchedulingStrategy::kFixedBlockSize,
        absl::nullopt, 1);

    // Parallelize where num_threads = num_shards. Very important!
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        num_threads, scheduling_params, fn);

    return partial_results;
  }

  template <typename Function, typename Op, typename... Args>
  typename Op::result_type RunReduce(uint64_t size, Function&& func, Op&& op,
                                     Args&&... args) const {
    auto partial_results = RunReduceP(size, func, std::move(op), args...);

    typename Op::result_type result = 0;

    for (auto partial_result : partial_results) {
      result = op(result, partial_result);
    }

    return result;
  }
};

// bad style standards here that we are forced to follow from qsim.
// computes the expectation value <state | p_sum | state > using
// scratch to save on memory. Implementation does this:
// 1. Copy state onto scratch
// 2. Evolve scratch forward with p_sum terms
// 3. Compute < state | scratch >
// 4. Sum and repeat.
// scratch is required to have memory initialized, but does not require
// values in memory to be set.
template <typename SimT, typename StateSpaceT, typename StateT>
tensorflow::Status ComputeExpectationQsim(const tfq::proto::PauliSum& p_sum,
                                          const SimT& sim,
                                          const StateSpaceT& ss, StateT& state,
                                          StateT& scratch,
                                          float* expectation_value) {
  // apply the  gates of the pauliterms to a copy of the wavefunction
  // and add up expectation value term by term.
  tensorflow::Status status = tensorflow::Status::OK();
  for (const tfq::proto::PauliTerm& term : p_sum.terms()) {
    // catch identity terms
    if (term.paulis_size() == 0) {
      *expectation_value += term.coefficient_real();
      // TODO(zaqqwerty): error somewhere if identities have any imaginary part
      continue;
    }

    QsimCircuit main_circuit;
    std::vector<qsim::GateFused<QsimGate>> fused_circuit;

    status = QsimCircuitFromPauliTerm(term, ss.num_qubits_, &main_circuit,
                                      &fused_circuit);

    if (!status.ok()) {
      return status;
    }
    // copy from src to scratch.
    ss.CopyState(state, scratch);
    for (int j = 0; j < fused_circuit.size(); j++) {
      qsim::ApplyFusedGate(sim, fused_circuit[j], scratch);
    }

    if (!status.ok()) {
      return status;
    }
    *expectation_value +=
        term.coefficient_real() * ss.RealInnerProduct(state, scratch);
  }
  return status;
}

// bad style standards here that we are forced to follow from qsim.
// computes the expectation value <state | p_sum | state > using
// scratch to save on memory. Implementation does this:
// 1. Copy state onto scratch
// 2. Convert scratch to Z basis
// 3. Compute < state | scratch > via sampling.
// 4. Sum and repeat.
// scratch is required to have memory initialized, but does not require
// values in memory to be set.
template <typename SimT, typename StateSpaceT, typename StateT>
tensorflow::Status ComputeSampledExpectationQsim(
    const tfq::proto::PauliSum& p_sum, const SimT& sim, const StateSpaceT& ss,
    StateT& state, StateT& scratch, const int num_samples,
    float* expectation_value) {
  if (num_samples == 0) {
    return tensorflow::Status::OK();
  }
  // apply the  gates of the pauliterms to a copy of the wavefunction
  // and add up expectation value term by term.
  tensorflow::Status status = tensorflow::Status::OK();
  for (const tfq::proto::PauliTerm& term : p_sum.terms()) {
    // catch identity terms
    if (term.paulis_size() == 0) {
      *expectation_value += term.coefficient_real();
      // TODO(zaqqwerty): error somewhere if identities have any imaginary part
      continue;
    }

    // Transform state into the measurement basis and sample it
    QsimCircuit main_circuit;
    std::vector<qsim::GateFused<QsimGate>> fused_circuit;

    status = QsimZBasisCircuitFromPauliTerm(term, ss.num_qubits_, &main_circuit,
                                            &fused_circuit);
    if (!status.ok()) {
      return status;
    }
    // copy from src to scratch.
    ss.CopyState(state, scratch);
    for (int j = 0; j < fused_circuit.size(); j++) {
      qsim::ApplyFusedGate(sim, fused_circuit[j], scratch);
    }

    if (!status.ok()) {
      return status;
    }

    const int seed = 1234;
    std::vector<uint64_t> state_samples = ss.Sample(scratch, num_samples, seed);

    // Find qubits on which to measure parity
    std::vector<unsigned int> parity_bits;
    for (const tfq::proto::PauliQubitPair& pair : term.paulis()) {
      unsigned int location;
      // GridQubit id should be parsed down to integer at this upstream
      //  so it is safe to just use atoi.
      bool unused = absl::SimpleAtoi(pair.qubit_id(), &location);
      // Parity functions use little-endian indexing
      parity_bits.push_back(ss.num_qubits_ - location - 1);
    }

    // Compute the BitMask.
    uint64_t mask = 0;
    for (int i = 0; i < parity_bits.size(); i++) {
      mask |= uint64_t(1) << uint64_t(parity_bits[i]);
    }

    // Compute the running parity.
    int parity_total(0);
    int count = 0;
    for (int i = 0; i < state_samples.size(); i++) {
      count = std::bitset<64>(state_samples[i] & mask).count() & 1;
      parity_total += count ? -1 : 1;
    }
    *expectation_value += static_cast<float>(parity_total) *
                          term.coefficient_real() /
                          static_cast<float>(num_samples);
  }
  return status;
}

}  // namespace tfq

#endif  // UTIL_QSIM_H_
