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

#include <memory>
#include <vector>

#include "../qsim/lib/channel.h"
#include "../qsim/lib/channels_cirq.h"
#include "../qsim/lib/circuit.h"
#include "../qsim/lib/circuit_noisy.h"
#include "../qsim/lib/fuser_mqubit.h"
#include "../qsim/lib/gate_appl.h"
#include "../qsim/lib/gates_cirq.h"
#include "../qsim/lib/io.h"
#include "../qsim/lib/qtrajectory.h"
#include "../qsim/lib/seqfor.h"
#include "../qsim/lib/simmux.h"
#include "cirq/google/api/v2/program.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow_quantum/core/ops/parse_context.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/src/util_qsim.h"

namespace tfq {

using ::cirq::google::api::v2::Program;
using ::tensorflow::Status;
using ::tfq::proto::PauliSum;

typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;
typedef qsim::NoisyCircuit<QsimGate> NoisyQsimCircuit;

class TfqNoisyExpectationOp : public tensorflow::OpKernel {
 public:
  explicit TfqNoisyExpectationOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // TODO (mbbrough): add more dimension checks for other inputs here.
    const int num_inputs = context->num_inputs();
    OP_REQUIRES(context, num_inputs == 5,
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Expected 5 inputs, got ", num_inputs, " inputs.")));

    // Create the output Tensor.
    const int output_dim_batch_size = context->input(0).dim_size(0);
    const int output_dim_op_size = context->input(3).dim_size(1);
    tensorflow::TensorShape output_shape;
    output_shape.AddDim(output_dim_batch_size);
    output_shape.AddDim(output_dim_op_size);

    tensorflow::Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_tensor = output->matrix<float>();

    std::vector<Program> programs;
    std::vector<int> num_qubits;
    std::vector<std::vector<PauliSum>> pauli_sums;
    OP_REQUIRES_OK(context, GetProgramsAndNumQubits(context, &programs,
                                                    &num_qubits, &pauli_sums));

    std::vector<SymbolMap> maps;
    OP_REQUIRES_OK(context, GetSymbolMaps(context, &maps));

    OP_REQUIRES(context, programs.size() == maps.size(),
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Number of circuits and symbol_values do not match. Got ",
                    programs.size(), " circuits and ", maps.size(),
                    " symbol values.")));

    std::vector<std::vector<int>> num_samples;
    OP_REQUIRES_OK(context, GetNumSamples(context, &num_samples));

    OP_REQUIRES(context, num_samples.size() == pauli_sums.size(),
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Dimension 0 of num_samples and pauli_sums do not match.",
                    "Got ", num_samples.size(), " lists of sample sizes and ",
                    pauli_sums.size(), " lists of pauli sums.")));

    OP_REQUIRES(
        context, context->input(4).dim_size(1) == context->input(3).dim_size(1),
        tensorflow::errors::InvalidArgument(absl::StrCat(
            "Dimension 1 of num_samples and pauli_sums do not match.", "Got ",
            context->input(4).dim_size(1), " lists of sample sizes and ",
            context->input(3).dim_size(1), " lists of pauli sums.")));

    // Construct qsim circuits.
    std::vector<NoisyQsimCircuit> qsim_circuits(programs.size(),
                                                NoisyQsimCircuit());

    auto construct_f = [&](int start, int end) {
      for (int i = start; i < end; i++) {
        OP_REQUIRES_OK(context, NoisyQsimCircuitFromProgram(
                                    programs[i], maps[i], num_qubits[i],
                                    &qsim_circuits[i]));
      }
    };

    const int num_cycles = 1000;
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        programs.size(), num_cycles, construct_f);

    int max_num_qubits = 0;
    for (const int num : num_qubits) {
      max_num_qubits = std::max(max_num_qubits, num);
    }

    // Cross reference with standard google cloud compute instances
    // Memory ~= 2 * num_threads * (2 * 64 * 2 ** num_qubits in circuits)
    // e2s2 = 2 CPU, 8GB -> Can safely do 25 since Memory = 4GB
    // e2s4 = 4 CPU, 16GB -> Can safely do 25 since Memory = 8GB
    // ...
    if (max_num_qubits >= 25) {
      // If the number of qubits is lager than 24, we switch to an
      // alternate parallelization scheme with runtime:
      // O(n_circuits * max_j(num_samples[i])) with parallelization being
      // multiple threads per wavefunction.
      ComputeLarge(num_qubits, qsim_circuits, pauli_sums, num_samples, context,
                   &output_tensor);
    } else {
      // Runtime: O(n_circuits * max_j(num_samples[i])) with parallelization
      // being done over number of trials.
      ComputeSmall(num_qubits, max_num_qubits, qsim_circuits, pauli_sums,
                   num_samples, context, &output_tensor);
    }
  }

 private:
  void ComputeLarge(const std::vector<int>& num_qubits,
                    const std::vector<NoisyQsimCircuit>& ncircuits,
                    const std::vector<std::vector<PauliSum>>& pauli_sums,
                    const std::vector<std::vector<int>>& num_samples,
                    tensorflow::OpKernelContext* context,
                    tensorflow::TTypes<float, 1>::Matrix* output_tensor) {
    // Instantiate qsim objects.
    const auto tfq_for = tfq::QsimFor(context);
    using Simulator = qsim::Simulator<const tfq::QsimFor&>;
    using StateSpace = Simulator::StateSpace;
    using QTSimulator =
        qsim::QuantumTrajectorySimulator<qsim::IO, QsimGate,
                                         qsim::MultiQubitGateFuser, Simulator>;

    // Begin simulation.
    int largest_nq = 1;
    Simulator sim = Simulator(tfq_for);
    StateSpace ss = StateSpace(tfq_for);
    auto sv = ss.Create(largest_nq);
    auto scratch = ss.Create(largest_nq);

    // Simulate programs one by one. Parallelizing over state vectors
    // we no longer parallelize over circuits. Each time we encounter a
    // a larger circuit we will grow the Statevector as necessary.
    for (int i = 0; i < ncircuits.size(); i++) {
      int nq = num_qubits[i];

      // (#679) Just ignore empty program
      if (ncircuits[i].channels.size() == 0) {
        for (int j = 0; j < pauli_sums[i].size(); j++) {
          (*output_tensor)(i, j) = -2.0;
        }
        continue;
      }

      if (nq > largest_nq) {
        largest_nq = nq;
        sv = ss.Create(largest_nq);
        scratch = ss.Create(largest_nq);
      }
      QTSimulator::Parameter param;
      param.collect_kop_stat = false;
      param.collect_mea_stat = false;
      param.normalize_before_mea_gates = true;
      std::vector<uint64_t> unused_stats;
      // Track op-wise stats.
      std::vector<int> run_samples(num_samples[i].size(), 0);
      std::vector<double> rolling_sums(num_samples[i].size(), 0.0);

      while (1) {
        ss.SetStateZero(sv);
        // time since epoch seeds random generator.
        unsigned long r_seed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
        QTSimulator::RunOnce(param, ncircuits[i], r_seed, ss, sim, scratch, sv,
                             unused_stats);

        // Use this trajectory as a source for all expectation calculations.
        for (int j = 0; j < pauli_sums[i].size(); j++) {
          if (run_samples[j] >= num_samples[i][j]) {
            continue;
          }
          float exp_v = 0.0;
          OP_REQUIRES_OK(context,
                         ComputeExpectationQsim(pauli_sums[i][j], sim, ss, sv,
                                                scratch, &exp_v));
          rolling_sums[j] += static_cast<double>(exp_v);
          run_samples[j]++;
        }
        bool break_loop = true;
        for (int j = 0; j < num_samples[i].size(); j++) {
          if (run_samples[j] < num_samples[i][j]) {
            break_loop = false;
            break;
          }
        }
        if (break_loop) {
          for (int j = 0; j < num_samples[i].size(); j++) {
            rolling_sums[j] /= num_samples[i][j];
            (*output_tensor)(i, j) = static_cast<float>(rolling_sums[j]);
          }
          break;
        }
      }
    }
  }

  void ComputeSmall(const std::vector<int>& num_qubits,
                    const int max_num_qubits,
                    const std::vector<NoisyQsimCircuit>& ncircuits,
                    const std::vector<std::vector<PauliSum>>& pauli_sums,
                    const std::vector<std::vector<int>>& num_samples,
                    tensorflow::OpKernelContext* context,
                    tensorflow::TTypes<float, 1>::Matrix* output_tensor) {
    using Simulator = qsim::Simulator<const qsim::SequentialFor&>;
    using StateSpace = Simulator::StateSpace;
    using QTSimulator =
        qsim::QuantumTrajectorySimulator<qsim::IO, QsimGate,
                                         qsim::MultiQubitGateFuser, Simulator>;

    const int output_dim_batch_size = output_tensor->dimension(0);
    const int output_dim_op_size = output_tensor->dimension(1);

    std::vector<tensorflow::mutex> batch_locks(output_dim_batch_size,
                                               tensorflow::mutex());

    const int num_threads = context->device()
                                ->tensorflow_cpu_worker_threads()
                                ->workers->NumThreads();
    output_tensor->setZero();
    auto DoWork = [&](int start, int end) {
      // Begin simulation.
      const auto tfq_for = qsim::SequentialFor(1);
      int largest_nq = 1;
      Simulator sim = Simulator(tfq_for);
      StateSpace ss = StateSpace(tfq_for);
      auto sv = ss.Create(largest_nq);
      auto scratch = ss.Create(largest_nq);

      for (int i = 0; i < ncircuits.size(); i++) {
        int nq = num_qubits[i];

        // (#679) Just ignore empty program
        if (ncircuits[i].channels.size() == 0) {
          for (int j = 0; j < pauli_sums[i].size(); j++) {
            (*output_tensor)(i, j) = -2.0;
          }
          continue;
        }

        if (nq > largest_nq) {
          largest_nq = nq;
          sv = ss.Create(largest_nq);
          scratch = ss.Create(largest_nq);
        }
        QTSimulator::Parameter param;
        param.collect_kop_stat = false;
        param.collect_mea_stat = false;
        param.normalize_before_mea_gates = true;
        std::vector<uint64_t> unused_stats;
        // Track op-wise stats.
        std::vector<int> run_samples(num_samples[i].size(), 0);
        std::vector<double> rolling_sums(num_samples[i].size(), 0.0);

        while (1) {
          ss.SetStateZero(sv);
          // time since epoch seeds random generator.
          unsigned long r_seed =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();

          QTSimulator::RunOnce(param, ncircuits[i], r_seed, ss, sim, scratch,
                               sv, unused_stats);

          // Compute expectations across all ops using this trajectory.
          for (int j = 0; j < pauli_sums[i].size(); j++) {
            if (run_samples[j] >=
                (num_samples[i][j] + num_threads - 1) / num_threads) {
              continue;
            }
            float exp_v = 0.0;
            OP_REQUIRES_OK(context,
                           ComputeExpectationQsim(pauli_sums[i][j], sim, ss, sv,
                                                  scratch, &exp_v));
            rolling_sums[j] += static_cast<double>(exp_v);
            run_samples[j]++;
          }

          // Check if we have run enough trajectories for all ops.
          bool break_loop = true;
          for (int j = 0; j < num_samples[i].size(); j++) {
            if (run_samples[j] <
                (num_samples[i][j] + num_threads - 1) / num_threads) {
              break_loop = false;
              break;
            }
          }
          if (break_loop) {
            // Lock writing to this batch index in output_tensor.
            batch_locks[i].lock();
            for (int j = 0; j < num_samples[i].size(); j++) {
              int divisor =
                  ((num_samples[i][j] + num_threads - 1) / num_threads) *
                  num_threads;
              rolling_sums[j] /= divisor;
              (*output_tensor)(i, j) += static_cast<float>(rolling_sums[j]);
            }
            batch_locks[i].unlock();
            break;
          }
        }
      }
    };

    // block_size = 1.
    tensorflow::thread::ThreadPool::SchedulingParams scheduling_params(
        tensorflow::thread::ThreadPool::SchedulingStrategy::kFixedBlockSize,
        absl::nullopt, 1);
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        num_threads, scheduling_params, DoWork);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TfqNoisyExpectation").Device(tensorflow::DEVICE_CPU),
    TfqNoisyExpectationOp);

REGISTER_OP("TfqNoisyExpectation")
    .Input("programs: string")
    .Input("symbol_names: string")
    .Input("symbol_values: float")
    .Input("pauli_sums: string")
    .Input("num_samples: int32")
    .Output("expectations: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle programs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &programs_shape));

      tensorflow::shape_inference::ShapeHandle symbol_names_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &symbol_names_shape));

      tensorflow::shape_inference::ShapeHandle symbol_values_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &symbol_values_shape));

      tensorflow::shape_inference::ShapeHandle pauli_sums_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &pauli_sums_shape));

      tensorflow::shape_inference::ShapeHandle num_samples_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &num_samples_shape));

      tensorflow::shape_inference::DimensionHandle output_rows =
          c->Dim(programs_shape, 0);
      tensorflow::shape_inference::DimensionHandle output_cols =
          c->Dim(pauli_sums_shape, 1);
      c->set_output(0, c->Matrix(output_rows, output_cols));

      return tensorflow::Status::OK();
    });

}  // namespace tfq
