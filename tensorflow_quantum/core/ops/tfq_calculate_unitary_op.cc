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

#include <string>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/gate_appl.h"
#include "../qsim/lib/gates_cirq.h"
#include "../qsim/lib/umux.h"
#include "cirq/google/api/v2/program.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow_quantum/core/ops/parse_context.h"
#include "tensorflow_quantum/core/src/circuit_parser_qsim.h"
#include "tensorflow_quantum/core/src/util_qsim.h"

namespace tfq {

using ::cirq::google::api::v2::Program;
using ::tensorflow::Status;

typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;

class TfqCalculateUnitaryOp : public tensorflow::OpKernel {
 public:
  explicit TfqCalculateUnitaryOp(tensorflow::OpKernelConstruction *context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext *context) override {
    // TODO (mbbrough): add more dimension checks for other inputs here.
    DCHECK_EQ(3, context->num_inputs());

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

    // TODO(pmassey): Investigate creating a matrix that isn't just the maximum
    // required size.
    const int output_dim_size = maps.size();
    tensorflow::TensorShape output_shape;
    output_shape.AddDim(output_dim_size);
    output_shape.AddDim(1 << max_num_qubits);
    output_shape.AddDim(1 << max_num_qubits);

    tensorflow::Tensor *output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_tensor = output->tensor<std::complex<float>, 3>();

    // Instantiate qsim objects.
    const auto tfq_for = tfq::QsimFor(context);
    using UCalculator = qsim::unitary::UnitaryCalculator<const tfq::QsimFor &>;
    using UnitarySpace = UCalculator::UnitarySpace;
    using Unitary = UnitarySpace::Unitary;

    // Begin simulation.
    int largest_nq = 1;
    Unitary u = UnitarySpace(tfq_for).CreateUnitary(largest_nq);

    // Simulate programs one by one. Parallelizing over state vectors
    // we no longer parallelize over circuits. Each time we encounter a
    // a larger circuit we will grow the unitary as nescessary.
    for (int i = 0; i < fused_circuits.size(); i++) {
      int nq = num_qubits[i];
      UCalculator sim = UCalculator(tfq_for);
      UnitarySpace us = UnitarySpace(tfq_for);
      if (nq > largest_nq) {
        // need to switch to larger unitaryspace.
        largest_nq = nq;
        u = us.CreateUnitary(nq);
      }
      us.SetIdentity(u);
      for (int j = 0; j < fused_circuits[i].size(); j++) {
        qsim::ApplyFusedGate(sim, fused_circuits[i][j], u);
      }

      // Parallel copy unitary information from qsim into tensorflow
      // tensors.
      auto copy_f = [i, nq, max_num_qubits, &output_tensor, &us, &u](
                        uint64_t start, uint64_t end) {
        uint64_t crossover = uint64_t(1) << nq;

        for (uint64_t l = start; l < end; l++) {
          uint64_t j = l / (1 << max_num_qubits);
          uint64_t k = l % (1 << max_num_qubits);
          if (k < crossover && j < crossover) {
            output_tensor(static_cast<ptrdiff_t>(i), static_cast<ptrdiff_t>(j),
                          static_cast<ptrdiff_t>(k)) = us.GetEntry(u, k, j);
          } else {
            output_tensor(static_cast<ptrdiff_t>(i), static_cast<ptrdiff_t>(j),
                          static_cast<ptrdiff_t>(k)) =
                std::complex<float>(-2, 0);
          }
        }
      };
      const uint64_t num_cycles_copy = 10 * (1 << max_num_qubits);
      context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
          (uint64_t(1) << max_num_qubits) * (uint64_t(1) << max_num_qubits),
          num_cycles_copy, copy_f);
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TfqCalculateUnitary").Device(tensorflow::DEVICE_CPU),
    TfqCalculateUnitaryOp);

REGISTER_OP("TfqCalculateUnitary")
    .Input("programs: string")
    .Input("symbol_names: string")
    .Input("symbol_values: float")
    .Output("unitary: complex64")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
      tensorflow::shape_inference::ShapeHandle programs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &programs_shape));

      tensorflow::shape_inference::ShapeHandle symbol_names_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &symbol_names_shape));

      tensorflow::shape_inference::ShapeHandle symbol_values_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &symbol_values_shape));

      c->set_output(
          0, c->MakeShape(
                 {c->Dim(programs_shape, 0),
                  tensorflow::shape_inference::InferenceContext::kUnknownDim,
                  tensorflow::shape_inference::InferenceContext::kUnknownDim}));

      return tensorflow::Status::OK();
    });

}  // namespace tfq
