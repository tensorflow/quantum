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

#include <custatevec.h>
#include <stdlib.h>

#include <chrono>
#include <string>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/gate_appl.h"
#include "../qsim/lib/gates_cirq.h"
#include "../qsim/lib/seqfor.h"
#include "../qsim/lib/simmux.h"
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
#include "tensorflow_quantum/core/proto/program.pb.h"
#include "tensorflow_quantum/core/src/circuit_parser_qsim.h"
#include "tensorflow_quantum/core/src/util_qsim.h"

namespace tfq {

using ::tensorflow::Status;
using ::tfq::proto::Program;

typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;

class TfqSimulateSamplesOpCuQuantum : public tensorflow::OpKernel {
 public:
  explicit TfqSimulateSamplesOpCuQuantum(
      tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
        OP_REQUIRES_OK(context, random_gen_.Init(context));
        // Allocates handlers for initialization.
        cublasCreate(&cublas_handle_);
        custatevecCreate(&custatevec_handle_);
      }

  ~TfqSimulateSamplesOpCuQuantum() {
      // Destroys handlers in sync with simulator lifetime.
      cublasDestroy(cublas_handle_);
      custatevecDestroy(custatevec_handle_);
    }

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

    Status parse_status = ::tensorflow::Status();
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

    ComputeLarge(num_qubits, max_num_qubits, num_samples, fused_circuits,
                 context, &output_tensor);
  }

 private:
  cublasHandle_t cublas_handle_;
  custatevecHandle_t custatevec_handle_;
  tensorflow::GuardedPhiloxRandom random_gen_;

  void ComputeLarge(
      const std::vector<int>& num_qubits, const int max_num_qubits,
      const int num_samples,
      const std::vector<std::vector<qsim::GateFused<QsimGate>>>& fused_circuits,
      tensorflow::OpKernelContext* context,
      tensorflow::TTypes<int8_t, 3>::Tensor* output_tensor) {
    // Instantiate qsim objects.
    using Simulator = qsim::SimulatorCuStateVec<float>;
    using StateSpace = Simulator::StateSpace;

    // Begin simulation.
    int largest_nq = 1;
    Simulator sim = Simulator(cublas_handle_, custatevec_handle_);
    StateSpace ss = StateSpace(cublas_handle_, custatevec_handle_);
    auto sv = ss.Create(largest_nq);

    auto local_gen = random_gen_.ReserveSamples32(fused_circuits.size() + 1);
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
};

REGISTER_KERNEL_BUILDER(
    Name("TfqSimulateSamplesCuquantum").Device(tensorflow::DEVICE_CPU),
    TfqSimulateSamplesOpCuQuantum);

REGISTER_OP("TfqSimulateSamplesCuquantum")
    .Input("programs: string")
    .Input("symbol_names: string")
    .Input("symbol_values: float")
    .Input("num_samples: int32")
    .SetIsStateful()
    .Output("samples: int8")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
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

      return ::tensorflow::Status();
    });

}  // namespace tfq