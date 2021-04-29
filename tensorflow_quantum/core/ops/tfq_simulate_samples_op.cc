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

#include <stdlib.h>

#include <string>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/gate_appl.h"
#include "../qsim/lib/gates_cirq.h"
#include "../qsim/lib/seqfor.h"
#include "../qsim/lib/simmux.h"
#include "cirq/google/api/v2/program.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow_quantum/core/ops/parse_context.h"
#include "tensorflow_quantum/core/src/circuit_parser_qsim.h"
#include "tensorflow_quantum/core/src/util_qsim.h"

namespace tfq {

using ::cirq::google::api::v2::Program;
using ::tensorflow::Status;

typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;

class TfqSimulateSamplesOp : public tensorflow::OpKernel {
 public:
  explicit TfqSimulateSamplesOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // TODO (mbbrough): add more dimension checks for other inputs here.
    DCHECK_EQ(4, context->num_inputs());

    // Parse to Program Proto and num_qubits.
    std::vector<Program> programs;
    std::vector<int> num_qubits;
    OP_REQUIRES_OK(context,
                   GetProgramsAndNumQubits(context, &programs, &num_qubits));

    // Parse symbol maps for parameter resolution in the circuits.
    std::vector<SymbolMap> maps;
    OP_REQUIRES_OK(context, GetSymbolMaps(context, &maps));
    OP_REQUIRES(
        context, maps.size() == programs.size(),
        tensorflow::errors::InvalidArgument(absl::StrCat(
            "Number of circuits and values do not match. Got ", programs.size(),
            " circuits and ", maps.size(), " values.")));

    int num_samples = 0;
    OP_REQUIRES_OK(context, GetIndividualSample(context, &num_samples));

    // Construct qsim circuits.
    std::vector<QsimCircuit> qsim_circuits(programs.size(), QsimCircuit());
    std::vector<std::vector<qsim::GateFused<QsimGate>>> fused_circuits(
        programs.size(), std::vector<qsim::GateFused<QsimGate>>({}));

    Status parse_status = Status::OK();
    auto p_lock = tensorflow::mutex();
    auto construct_f = [&](int start, int end) {
      for (int i = start; i < end; i++) {
        Status local =
            QsimCircuitFromProgram(programs[i], maps[i], num_qubits[i],
                                   &qsim_circuits[i], &fused_circuits[i]);
        NESTED_FN_STATUS_SYNC(parse_status, local, p_lock);
      }
    };

    const int num_cycles = 1000;
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        programs.size(), num_cycles, construct_f);
    OP_REQUIRES_OK(context, parse_status);

    // Find largest circuit for tensor size padding and allocate
    // the output tensor.
    int max_num_qubits = 0;
    for (const int num : num_qubits) {
      max_num_qubits = std::max(max_num_qubits, num);
    }

    const int output_dim_size = maps.size();
    tensorflow::TensorShape output_shape;
    output_shape.AddDim(output_dim_size);
    output_shape.AddDim(num_samples);
    output_shape.AddDim(max_num_qubits);

    tensorflow::Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_tensor = output->tensor<int8_t, 3>();

    if (num_samples == 0) {
      return;  // bug in qsim dependency we can't control.
    }

    // Cross reference with standard google cloud compute instances
    // Memory ~= 2 * num_threads * (2 * 64 * 2 ** num_qubits in circuits)
    // e2s2 = 2 CPU, 8GB -> Can safely do 25 since Memory = 4GB
    // e2s4 = 4 CPU, 16GB -> Can safely do 25 since Memory = 8GB
    // ...
    if (max_num_qubits >= 26 || programs.size() == 1) {
      ComputeLarge(num_qubits, max_num_qubits, num_samples, fused_circuits,
                   context, &output_tensor);
    } else {
      ComputeSmall(num_qubits, max_num_qubits, num_samples, fused_circuits,
                   context, &output_tensor);
    }
  }

 private:
  void ComputeLarge(
      const std::vector<int>& num_qubits, const int max_num_qubits,
      const int num_samples,
      const std::vector<std::vector<qsim::GateFused<QsimGate>>>& fused_circuits,
      tensorflow::OpKernelContext* context,
      tensorflow::TTypes<int8_t, 3>::Tensor* output_tensor) {
    // Instantiate qsim objects.
    const auto tfq_for = tfq::QsimFor(context);
    using Simulator = qsim::Simulator<const tfq::QsimFor&>;
    using StateSpace = Simulator::StateSpace;

    // Begin simulation.
    int largest_nq = 1;
    Simulator sim = Simulator(tfq_for);
    StateSpace ss = StateSpace(tfq_for);
    auto sv = ss.Create(largest_nq);

    tensorflow::GuardedPhiloxRandom random_gen;
    random_gen.Init(tensorflow::random::New64(), tensorflow::random::New64());
    auto local_gen = random_gen.ReserveSamples32(fused_circuits.size() + 1);
    tensorflow::random::SimplePhilox rand_source(&local_gen);

    // Simulate programs one by one. Parallelizing over state vectors
    // we no longer parallelize over circuits. Each time we encounter a
    // a larger circuit we will grow the Statevector as nescessary.
    for (int i = 0; i < fused_circuits.size(); i++) {
      int nq = num_qubits[i];

      if (nq > largest_nq) {
        // need to switch to larger statespace.
        largest_nq = nq;
        sv = ss.Create(largest_nq);
      }
      ss.SetStateZero(sv);
      for (int j = 0; j < fused_circuits[i].size(); j++) {
        qsim::ApplyFusedGate(sim, fused_circuits[i][j], sv);
      }

      auto samples = ss.Sample(sv, num_samples, rand_source.Rand32());
      for (int j = 0; j < num_samples; j++) {
        uint64_t q_ind = 0;
        uint64_t mask = 1;
        bool val = 0;
        while (q_ind < nq) {
          val = samples[j] & mask;
          (*output_tensor)(
              i, j, static_cast<ptrdiff_t>(max_num_qubits - q_ind - 1)) = val;
          q_ind++;
          mask <<= 1;
        }
        while (q_ind < max_num_qubits) {
          (*output_tensor)(
              i, j, static_cast<ptrdiff_t>(max_num_qubits - q_ind - 1)) = -2;
          q_ind++;
        }
      }
    }
  }

  void ComputeSmall(
      const std::vector<int>& num_qubits, const int max_num_qubits,
      const int num_samples,
      const std::vector<std::vector<qsim::GateFused<QsimGate>>>& fused_circuits,
      tensorflow::OpKernelContext* context,
      tensorflow::TTypes<int8_t, 3>::Tensor* output_tensor) {
    const auto tfq_for = qsim::SequentialFor(1);
    using Simulator = qsim::Simulator<const qsim::SequentialFor&>;
    using StateSpace = Simulator::StateSpace;

    tensorflow::GuardedPhiloxRandom random_gen;
    random_gen.Init(tensorflow::random::New64(), tensorflow::random::New64());

    auto DoWork = [&](int start, int end) {
      int largest_nq = 1;
      Simulator sim = Simulator(tfq_for);
      StateSpace ss = StateSpace(tfq_for);
      auto sv = ss.Create(largest_nq);

      auto local_gen = random_gen.ReserveSamples32(fused_circuits.size() + 1);
      tensorflow::random::SimplePhilox rand_source(&local_gen);

      for (int i = start; i < end; i++) {
        int nq = num_qubits[i];

        if (nq > largest_nq) {
          // need to switch to larger statespace.
          largest_nq = nq;
          sv = ss.Create(largest_nq);
        }
        ss.SetStateZero(sv);
        for (int j = 0; j < fused_circuits[i].size(); j++) {
          qsim::ApplyFusedGate(sim, fused_circuits[i][j], sv);
        }

        auto samples = ss.Sample(sv, num_samples, rand_source.Rand32());
        for (int j = 0; j < num_samples; j++) {
          uint64_t q_ind = 0;
          uint64_t mask = 1;
          bool val = 0;
          while (q_ind < nq) {
            val = samples[j] & mask;
            (*output_tensor)(
                i, j, static_cast<ptrdiff_t>(max_num_qubits - q_ind - 1)) = val;
            q_ind++;
            mask <<= 1;
          }
          while (q_ind < max_num_qubits) {
            (*output_tensor)(
                i, j, static_cast<ptrdiff_t>(max_num_qubits - q_ind - 1)) = -2;
            q_ind++;
          }
        }
      }
    };

    const int64_t num_cycles =
        200 * (int64_t(1) << static_cast<int64_t>(max_num_qubits));
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        fused_circuits.size(), num_cycles, DoWork);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TfqSimulateSamples").Device(tensorflow::DEVICE_CPU),
    TfqSimulateSamplesOp);

REGISTER_OP("TfqSimulateSamples")
    .Input("programs: string")
    .Input("symbol_names: string")
    .Input("symbol_values: float")
    .Input("num_samples: int32")
    .Output("samples: int8")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle programs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &programs_shape));

      tensorflow::shape_inference::ShapeHandle symbol_names_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &symbol_names_shape));

      tensorflow::shape_inference::ShapeHandle symbol_values_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &symbol_values_shape));

      tensorflow::shape_inference::ShapeHandle num_samples_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &num_samples_shape));

      // [batch_size, n_samples, largest_n_qubits]
      c->set_output(
          0, c->MakeShape(
                 {c->Dim(programs_shape, 0),
                  tensorflow::shape_inference::InferenceContext::kUnknownDim,
                  tensorflow::shape_inference::InferenceContext::kUnknownDim}));

      return tensorflow::Status::OK();
    });

}  // namespace tfq
