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

  template <typename Function, typename Op, typename... Args>
  typename Op::result_type RunReduce(unsigned num_threads, uint64_t size,
                                     Function&& func, Op&& op,
                                     Args&&... args) const {
    // TODO(mbbrough): implement the rest of this for Expectation functions.
    return 0;
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

}  // namespace tfq

#endif  // UTIL_QSIM_H_
