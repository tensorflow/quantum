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
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/fuser.h"
#include "../qsim/lib/gate_appl.h"
#include "../qsim/lib/gates_cirq.h"
#include "../qsim/lib/matrix.h"
#include "../qsim/lib/mps_statespace.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/src/circuit_parser_qsim.h"

namespace tfq {

typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;
typedef std::vector<qsim::GateFused<QsimGate>> QsimFusedCircuit;

// Custom FOR loop struct to use TF threadpool instead of native
// qsim OpenMP or serial FOR implementations.
struct QsimFor {
  tensorflow::OpKernelContext* context;
  QsimFor(tensorflow::OpKernelContext* cxt) { context = cxt; }

  template <typename Function, typename... Args>
  void Run(uint64_t size, Function&& func, Args&&... args) const {
    auto worker_f = [&func, &args...](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
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
                                          float* expectation_value,
                                          bool fuse_paulis = true) {
  // apply the gates of the pauliterms to a copy of the state vector
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

    status = QsimCircuitFromPauliTerm(term, state.num_qubits(), &main_circuit,
                                      &fused_circuit);

    if (!status.ok()) {
      return status;
    }
    // copy from src to scratch.
    ss.Copy(state, scratch);
    if (fuse_paulis) {
      for (const qsim::GateFused<QsimGate>& fused_gate : fused_circuit) {
        qsim::ApplyFusedGate(sim, fused_gate, scratch);
      }
    } else {
      for (const auto& unfused_gate : main_circuit.gates) {
        qsim::ApplyGate(sim, unfused_gate, scratch);
      }
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
    tensorflow::random::SimplePhilox& random_source, float* expectation_value) {
  std::uniform_int_distribution<> distrib(1, 1 << 30);

  if (num_samples == 0) {
    return tensorflow::Status::OK();
  }
  // apply the gates of the pauliterms to a copy of the state vector
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

    status = QsimZBasisCircuitFromPauliTerm(term, state.num_qubits(),
                                            &main_circuit, &fused_circuit);
    if (!status.ok()) {
      return status;
    }
    // copy from src to scratch.
    ss.Copy(state, scratch);
    for (const qsim::GateFused<QsimGate>& fused_gate : fused_circuit) {
      qsim::ApplyFusedGate(sim, fused_gate, scratch);
    }

    if (!status.ok()) {
      return status;
    }
    std::vector<uint64_t> state_samples =
        ss.Sample(scratch, num_samples, random_source.Rand32());

    // Find qubits on which to measure parity
    std::vector<unsigned int> parity_bits;
    for (const tfq::proto::PauliQubitPair& pair : term.paulis()) {
      unsigned int location;
      // GridQubit id should be parsed down to integer at this upstream
      //  so it is safe to just use atoi.
      (void)absl::SimpleAtoi(pair.qubit_id(), &location);
      // Parity functions use little-endian indexing
      parity_bits.push_back(state.num_qubits() - location - 1);
    }

    // Compute the BitMask.
    uint64_t mask = 0;
    for (const unsigned int parity_bit : parity_bits) {
      mask |= uint64_t(1) << uint64_t(parity_bit);
    }

    // Compute the running parity.
    int parity_total(0);
    int count = 0;
    for (const uint64_t state_sample : state_samples) {
      count = std::bitset<64>(state_sample & mask).count() & 1;
      parity_total += count ? -1 : 1;
    }
    *expectation_value += static_cast<float>(parity_total) *
                          term.coefficient_real() /
                          static_cast<float>(num_samples);
  }
  return status;
}

// Overloading for MPS : it requires more scratch states.
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
    StateT& state, StateT& scratch, StateT& scratch2, StateT& scratch3,
    const int num_samples, tensorflow::random::SimplePhilox& random_source,
    float* expectation_value) {
  std::uniform_int_distribution<> distrib(1, 1 << 30);

  if (num_samples == 0) {
    return tensorflow::Status::OK();
  }
  // apply the gates of the pauliterms to a copy of the state vector
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

    status = QsimZBasisCircuitFromPauliTerm(term, state.num_qubits(),
                                            &main_circuit, &fused_circuit);
    if (!status.ok()) {
      return status;
    }
    // copy from src to scratch.
    ss.Copy(state, scratch);
    for (const auto& unfused_gate : main_circuit.gates) {
      qsim::ApplyGate(sim, unfused_gate, scratch);
    }

    if (!status.ok()) {
      return status;
    }
    std::vector<std::vector<bool>> state_samples;

    ss.Sample(scratch, scratch2, scratch3, num_samples, random_source.Rand32(),
              &state_samples);

    // Find qubits on which to measure parity and compute the BitMask.
    std::vector<bool> mask;
    const unsigned int max_num_qubits = state.num_qubits();
    mask.reserve(max_num_qubits);
    for (const tfq::proto::PauliQubitPair& pair : term.paulis()) {
      unsigned int location;
      // GridQubit id should be parsed down to integer at this upstream
      //  so it is safe to just use atoi.
      (void)absl::SimpleAtoi(pair.qubit_id(), &location);
      // Parity functions use little-endian indexing
      mask[max_num_qubits - location - 1] = 1;
    }

    // Compute the running parity.
    int parity_total(0);
    int count = 0;
    for (std::vector<bool>& state_sample : state_samples) {
      std::transform(mask.begin(), mask.end(), state_sample.begin(),
                     state_sample.begin(),
                     [](bool x, bool y) -> bool { return x & y; });
      count = std::accumulate(state_sample.begin(), state_sample.end(), 0);
      parity_total += (count & 1) ? -1 : 1;
    }
    *expectation_value += static_cast<float>(parity_total) *
                          term.coefficient_real() /
                          static_cast<float>(num_samples);
  }
  return status;
}

// Assumes p_sums.size() == op_coeffs.size()
// state stores |psi>. scratch has been created, but does not
// require initialization. dest has been created, but does not require
// initialization.
// After termination scratch will contain a copy of source.
template <typename SimT, typename StateSpaceT, typename StateT>
tensorflow::Status AccumulateOperators(
    const std::vector<tfq::proto::PauliSum>& p_sums,
    const std::vector<float>& op_coeffs, const SimT& sim, const StateSpaceT& ss,
    StateT& source, StateT& scratch, StateT& dest) {
  // apply the gates of the pauliterms to a copy of the state vector
  // accumulating results as we go. Effectively doing O|psi> for an arbitrary
  // O. Result is stored on scratch.
  tensorflow::Status status = tensorflow::Status::OK();
  ss.Copy(source, scratch);
  ss.SetAllZeros(dest);

  DCHECK_EQ(p_sums.size(), op_coeffs.size());

  for (size_t i = 0; i < p_sums.size(); i++) {
    for (const tfq::proto::PauliTerm& term : p_sums[i].terms()) {
      const float leading_coeff = op_coeffs[i] * term.coefficient_real();
      if (std::fabs(leading_coeff) < 1e-5) {
        // skip really small terms that will just induce more rounding
        // errors.
        continue;
      }
      if (term.paulis_size() == 0) {
        // identity term. Scalar multiply, add, then revert.
        ss.Multiply(leading_coeff, scratch);
        ss.Add(scratch, dest);
        ss.Copy(source, scratch);
        continue;
      }

      QsimCircuit main_circuit;
      std::vector<qsim::GateFused<QsimGate>> fused_circuit;

      status = QsimCircuitFromPauliTerm(term, source.num_qubits(),
                                        &main_circuit, &fused_circuit);
      if (!status.ok()) {
        return status;
      }

      // Apply scaled gates, accumulate, undo.
      for (const qsim::GateFused<QsimGate>& fused_gate : fused_circuit) {
        qsim::ApplyFusedGate(sim, fused_gate, scratch);
      }

      ss.Multiply(leading_coeff, scratch);
      ss.Add(scratch, dest);
      ss.Copy(source, scratch);
      // scratch should now be reverted back to original source.
    }
  }

  return status;
}

// Assumes coefficients.size() == fused_circuits.size().
// These are checked at the upstream.
// scratch has been created, but does not require initialization.
// dest has been created, but does not require initialization.
// scratch has garbage value.
// |psi> = sum_i coefficients[i]*|phi[i]>
template <typename SimT, typename StateSpaceT, typename StateT>
tensorflow::Status AccumulateFusedCircuits(
    const std::vector<float>& coefficients,
    const std::vector<QsimFusedCircuit>& fused_circuits, const SimT& sim,
    const StateSpaceT& ss, StateT& scratch, StateT& dest) {
  tensorflow::Status status = tensorflow::Status::OK();
  ss.SetAllZeros(dest);

  for (std::vector<qsim::GateFused<QsimGate>>::size_type i = 0;
       i < fused_circuits.size(); i++) {
    ss.SetStateZero(scratch);
    for (std::vector<qsim::GateFused<QsimGate>>::size_type j = 0;
         j < fused_circuits[i].size(); j++) {
      qsim::ApplyFusedGate(sim, fused_circuits[i][j], scratch);
    }
    ss.Multiply(coefficients[i], scratch);
    ss.Add(scratch, dest);
  }

  return status;
}

// Balance the number of trajectory computations done between
// threads. num_samples is a 2d vector containing the number of reps
// requested for each pauli_sum[i,j]. After running thread_offsets
// contains 0/-1 values that will offset the work for each thread.
// to make it as close to uniform as possible. **Assumes circuits
// have rouhgly equal simulation cost**
static void BalanceTrajectory(const std::vector<std::vector<int>>& num_samples,
                              const int& num_threads,
                              std::vector<std::vector<int>>* thread_offsets) {
  std::vector<int> rep_limits(num_samples.size(), -1);
  std::vector<int> height(num_threads, 0);

  for (int i = 0; i < num_samples.size(); i++) {
    for (int j = 0; j < num_samples[i].size(); j++) {
      rep_limits[i] = std::max(rep_limits[i], num_samples[i][j]);
    }
  }
  int prev_max_height = -1;
  for (int j = 0; j < num_samples.size(); j++) {
    int run_ceiling = ((rep_limits[j] + num_threads - 1) / num_threads);
    int num_lo = num_threads * run_ceiling - rep_limits[j];
    int num_hi = num_threads - num_lo;
    int cur_max = prev_max_height;
    for (int i = 0; i < num_threads; i++) {
      if (height[i] == cur_max && num_lo) {
        // previously had extra work on this thread and
        // have remaining low budget to give.
        height[i]++;
        (*thread_offsets)[i][j] = -1;
        num_lo--;
      } else if (height[i] == cur_max - 1 && num_hi) {
        // previously had less work on this thread and
        // remaining high budget to give.
        height[i] += 2;
        (*thread_offsets)[i][j] = 0;
        num_hi--;
      } else if (num_hi) {
        height[i] += 2;
        (*thread_offsets)[i][j] = 0;
        num_hi--;
      } else {
        height[i]++;
        (*thread_offsets)[i][j] = -1;
        num_lo--;
      }
      prev_max_height = std::max(height[i], prev_max_height);
    }
  }
}

// Simpler case of TrajectoryBalance where num_samples is fixed
// across all circuits.
static void BalanceTrajectory(const int& num_samples, const int& num_threads,
                              std::vector<std::vector<int>>* thread_offsets) {
  std::vector<int> height(num_threads, 0);

  int prev_max_height = -1;
  for (int j = 0; j < (*thread_offsets)[0].size(); j++) {
    int run_ceiling = ((num_samples + num_threads - 1) / num_threads);
    int num_lo = num_threads * run_ceiling - num_samples;
    int num_hi = num_threads - num_lo;
    int cur_max = prev_max_height;
    for (int i = 0; i < num_threads; i++) {
      if (height[i] == cur_max && num_lo) {
        // previously had extra work on this thread and
        // have remaining low budget to give.
        height[i]++;
        (*thread_offsets)[i][j] = -1;
        num_lo--;
      } else if (height[i] == cur_max - 1 && num_hi) {
        // previously had less work on this thread and
        // remaining high budget to give.
        height[i] += 2;
        (*thread_offsets)[i][j] = 0;
        num_hi--;
      } else if (num_hi) {
        height[i] += 2;
        (*thread_offsets)[i][j] = 0;
        num_hi--;
      } else {
        height[i]++;
        (*thread_offsets)[i][j] = -1;
        num_lo--;
      }
      prev_max_height = std::max(height[i], prev_max_height);
    }
  }
}

}  // namespace tfq

#endif  // UTIL_QSIM_H_
